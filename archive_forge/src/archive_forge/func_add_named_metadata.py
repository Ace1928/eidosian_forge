import collections
from llvmlite.ir import context, values, types, _utils
def add_named_metadata(self, name, element=None):
    """
        Add a named metadata node to the module, if it doesn't exist,
        or return the existing node.
        If *element* is given, it will append a new element to
        the named metadata node.  If *element* is a sequence of values
        (rather than a metadata value), a new unnamed node will first be
        created.

        Example::
            module.add_named_metadata("llvm.ident", ["llvmlite/1.0"])
        """
    if name in self.namedmetadata:
        nmd = self.namedmetadata[name]
    else:
        nmd = self.namedmetadata[name] = values.NamedMetaData(self)
    if element is not None:
        if not isinstance(element, values.Value):
            element = self.add_metadata(element)
        if not isinstance(element.type, types.MetaDataType):
            raise TypeError('wrong type for metadata element: got %r' % (element,))
        nmd.add(element)
    return nmd