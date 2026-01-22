from __future__ import (absolute_import, division, print_function)
class RestrictedModuleLoader:
    """Python module loader that restricts inappropriate imports."""

    def __init__(self, path, name, restrict_to_module_paths):
        self.path = path
        self.name = name
        self.loaded_modules = set()
        self.restrict_to_module_paths = restrict_to_module_paths

    def find_spec(self, fullname, path=None, target=None):
        """Return the spec from the loader or None"""
        loader = self._get_loader(fullname, path=path)
        if loader is not None:
            if has_py3_loader:
                return spec_from_loader(fullname, loader)
            raise ImportError("Failed to import '%s' due to a bug in ansible-test. Check importlib imports for typos." % fullname)
        return None

    def find_module(self, fullname, path=None):
        """Return self if the given fullname is restricted, otherwise return None."""
        return self._get_loader(fullname, path=path)

    def _get_loader(self, fullname, path=None):
        """Return self if the given fullname is restricted, otherwise return None."""
        if fullname in self.loaded_modules:
            return None
        if is_name_in_namepace(fullname, ['ansible']):
            if not self.restrict_to_module_paths:
                return None
            if fullname in ('ansible.module_utils.basic',):
                return self
            if is_name_in_namepace(fullname, ['ansible.module_utils', self.name]):
                return None
            if any((os.path.exists(candidate_path) for candidate_path in convert_ansible_name_to_absolute_paths(fullname))):
                return self
            return None
        if is_name_in_namepace(fullname, ['ansible_collections']):
            if not collection_loader:
                return self
            if not self.restrict_to_module_paths:
                return None
            if is_name_in_namepace(fullname, ['ansible_collections...plugins.module_utils', self.name]):
                return None
            if collection_loader.find_module(fullname, path):
                return self
            return None
        return None

    def create_module(self, spec):
        """Return None to use default module creation."""
        return None

    def exec_module(self, module):
        """Execute the module if the name is ansible.module_utils.basic and otherwise raise an ImportError"""
        fullname = module.__spec__.name
        if fullname == 'ansible.module_utils.basic':
            self.loaded_modules.add(fullname)
            for path in convert_ansible_name_to_absolute_paths(fullname):
                if not os.path.exists(path):
                    continue
                loader = SourceFileLoader(fullname, path)
                spec = spec_from_loader(fullname, loader)
                real_module = module_from_spec(spec)
                loader.exec_module(real_module)
                real_module.AnsibleModule = ImporterAnsibleModule
                real_module._load_params = lambda *args, **kwargs: {}
                sys.modules[fullname] = real_module
                return None
            raise ImportError('could not find "%s"' % fullname)
        raise ImportError('import of "%s" is not allowed in this context' % fullname)

    def load_module(self, fullname):
        """Return the module if the name is ansible.module_utils.basic and otherwise raise an ImportError."""
        if fullname == 'ansible.module_utils.basic':
            module = self.__load_module(fullname)
            module.AnsibleModule = ImporterAnsibleModule
            module._load_params = lambda *args, **kwargs: {}
            return module
        raise ImportError('import of "%s" is not allowed in this context' % fullname)

    def __load_module(self, fullname):
        """Load the requested module while avoiding infinite recursion."""
        self.loaded_modules.add(fullname)
        return import_module(fullname)