from ..common.utils import bytes2str
class TypeDesc(object):
    """ Encapsulates a description of a datatype, as parsed from DWARF DIEs.
        Not enough to display the variable in the debugger, but enough
        to produce a type description string similar to those of llvm-dwarfdump.

        name - name for primitive datatypes, element name for arrays, the
            whole name for functions and function pouinters

        modifiers - a collection of "const"/"pointer"/"reference", from the
            chain of DIEs preceeding the real type DIE

        scopes - a collection of struct/class/namespace names, parents of the
            real type DIE

        tag - the tag of the real type DIE, stripped of initial DW_TAG_ and
            final _type

        dimensions - the collection of array dimensions, if the type is an
            array. -1 means an array of unknown dimension.

    """

    def __init__(self):
        self.name = None
        self.modifiers = ()
        self.scopes = ()
        self.tag = None
        self.dimensions = None

    def __str__(self):
        name = str(self.name)
        mods = self.modifiers
        parts = []
        if len(mods) and mods[0] == 'const':
            parts.append('const')
            mods = mods[1:]
        if mods[-2:] == ('reference', 'const'):
            parts.append('const')
            mods = mods[0:-1]
        if self.scopes:
            name = '::'.join(self.scopes) + '::' + name
        parts.append(name)
        if len(mods):
            parts.append(''.join((cpp_symbols[mod] for mod in mods)))
        if self.dimensions:
            dims = ''.join(('[%s]' % (str(dim) if dim > 0 else '',) for dim in self.dimensions))
        else:
            dims = ''
        return ' '.join(parts) + dims