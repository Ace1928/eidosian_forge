import re
class PackageMangler:
    """
    Used on import, to ensure that all modules imported have a shared mangle parent.
    """

    def __init__(self):
        global _mangle_index
        self._mangle_index = _mangle_index
        _mangle_index += 1
        self._mangle_parent = f'<torch_package_{self._mangle_index}>'

    def mangle(self, name) -> str:
        assert len(name) != 0
        return self._mangle_parent + '.' + name

    def demangle(self, mangled: str) -> str:
        """
        Note: This only demangles names that were mangled by this specific
        PackageMangler. It will pass through names created by a different
        PackageMangler instance.
        """
        if mangled.startswith(self._mangle_parent + '.'):
            return mangled.partition('.')[2]
        return mangled

    def parent_name(self):
        return self._mangle_parent