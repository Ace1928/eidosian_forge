from suds import *
from suds.umx import *
from suds.umx.typed import Typed
from suds.sax import Namespace
def applyaty(self, content, xty):
    """
        Apply the type referenced in the I{arrayType} to the content
        (child nodes) of the array.  Each element (node) in the array
        that does not have an explicit xsi:type attribute is given one
        based on the I{arrayType}.
        @param content: An array content.
        @type content: L{Content}
        @param xty: The XSI type reference.
        @type xty: str
        @return: self
        @rtype: L{Encoded}
        """
    name = 'type'
    ns = Namespace.xsins
    parent = content.node
    for child in parent.getChildren():
        ref = child.get(name, ns)
        if ref is None:
            parent.addPrefix(ns[0], ns[1])
            attr = ':'.join((ns[0], name))
            child.set(attr, xty)
    return self