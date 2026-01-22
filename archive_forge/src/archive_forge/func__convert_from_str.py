from .errors import BzrError, InternalBzrError
def _convert_from_str(self, from_str):
    """This converts a 'from foo import bar' string into an import map.

        :param from_str: The import string to process
        """
    if not from_str.startswith('from '):
        raise ValueError('bad from/import %r' % from_str)
    from_str = from_str[len('from '):]
    from_module, import_list = from_str.split(' import ')
    from_module_path = from_module.split('.')
    if not from_module_path[0]:
        raise ImportError(from_module)
    for path in import_list.split(','):
        path = path.strip()
        if not path:
            continue
        as_hunks = path.split(' as ')
        if len(as_hunks) == 2:
            name = as_hunks[1].strip()
            module = as_hunks[0].strip()
        else:
            name = module = path
        if name in self.imports:
            raise ImportNameCollision(name)
        self.imports[name] = (from_module_path, module, {})