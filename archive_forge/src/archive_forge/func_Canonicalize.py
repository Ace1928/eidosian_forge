import string
from xml.dom import Node
def Canonicalize(node, output=None, **kw):
    """Canonicalize(node, output=None, **kw) -> UTF-8

    Canonicalize a DOM document/element node and all descendents.
    Return the text; if output is specified then output.write will
    be called to output the text and None will be returned
    Keyword parameters:
        nsdict: a dictionary of prefix:uri namespace entries
                assumed to exist in the surrounding context
        comments: keep comments if non-zero (default is 0)
        subset: Canonical XML subsetting resulting from XPath
                (default is [])
        unsuppressedPrefixes: do exclusive C14N, and this specifies the
                prefixes that should be inherited.
    """
    if output:
        apply(_implementation, (node, output.write), kw)
    else:
        s = StringIO.StringIO()
        apply(_implementation, (node, s.write), kw)
        return s.getvalue()