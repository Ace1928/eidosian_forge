from .errors import BzrError, InternalBzrError
def _convert_imports(self, scope):
    for name, info in self.imports.items():
        self._lazy_import_class(scope, name=name, module_path=info[0], member=info[1], children=info[2])