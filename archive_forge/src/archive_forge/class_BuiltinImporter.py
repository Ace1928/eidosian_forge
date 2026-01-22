class BuiltinImporter:
    """Meta path import for built-in modules.

    All methods are either class or static methods to avoid the need to
    instantiate the class.

    """
    _ORIGIN = 'built-in'

    @staticmethod
    def module_repr(module):
        """Return repr for the module.

        The method is deprecated.  The import machinery does the job itself.

        """
        _warnings.warn('BuiltinImporter.module_repr() is deprecated and slated for removal in Python 3.12', DeprecationWarning)
        return f'<module {module.__name__!r} ({BuiltinImporter._ORIGIN})>'

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if _imp.is_builtin(fullname):
            return spec_from_loader(fullname, cls, origin=cls._ORIGIN)
        else:
            return None

    @classmethod
    def find_module(cls, fullname, path=None):
        """Find the built-in module.

        If 'path' is ever specified then the search is considered a failure.

        This method is deprecated.  Use find_spec() instead.

        """
        _warnings.warn('BuiltinImporter.find_module() is deprecated and slated for removal in Python 3.12; use find_spec() instead', DeprecationWarning)
        spec = cls.find_spec(fullname, path)
        return spec.loader if spec is not None else None

    @staticmethod
    def create_module(spec):
        """Create a built-in module"""
        if spec.name not in sys.builtin_module_names:
            raise ImportError('{!r} is not a built-in module'.format(spec.name), name=spec.name)
        return _call_with_frames_removed(_imp.create_builtin, spec)

    @staticmethod
    def exec_module(module):
        """Exec a built-in module"""
        _call_with_frames_removed(_imp.exec_builtin, module)

    @classmethod
    @_requires_builtin
    def get_code(cls, fullname):
        """Return None as built-in modules do not have code objects."""
        return None

    @classmethod
    @_requires_builtin
    def get_source(cls, fullname):
        """Return None as built-in modules do not have source code."""
        return None

    @classmethod
    @_requires_builtin
    def is_package(cls, fullname):
        """Return False as built-in modules are never packages."""
        return False
    load_module = classmethod(_load_module_shim)