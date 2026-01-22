import _imp
import _io
import sys
import _warnings
import marshal
class NamespaceLoader:

    def __init__(self, name, path, path_finder):
        self._path = _NamespacePath(name, path, path_finder)

    @staticmethod
    def module_repr(module):
        """Return repr for the module.

        The method is deprecated.  The import machinery does the job itself.

        """
        _warnings.warn('NamespaceLoader.module_repr() is deprecated and slated for removal in Python 3.12', DeprecationWarning)
        return '<module {!r} (namespace)>'.format(module.__name__)

    def is_package(self, fullname):
        return True

    def get_source(self, fullname):
        return ''

    def get_code(self, fullname):
        return compile('', '<string>', 'exec', dont_inherit=True)

    def create_module(self, spec):
        """Use default semantics for module creation."""

    def exec_module(self, module):
        pass

    def load_module(self, fullname):
        """Load a namespace module.

        This method is deprecated.  Use exec_module() instead.

        """
        _bootstrap._verbose_message('namespace module loaded with path {!r}', self._path)
        return _bootstrap._load_module_shim(self, fullname)

    def get_resource_reader(self, module):
        from importlib.readers import NamespaceReader
        return NamespaceReader(self._path)