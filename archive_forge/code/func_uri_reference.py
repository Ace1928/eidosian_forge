from .iri import IRIReference
from .parseresult import ParseResult
from .uri import URIReference
def uri_reference(uri, encoding='utf-8'):
    """Parse a URI string into a URIReference.

    This is a convenience function. You could achieve the same end by using
    ``URIReference.from_string(uri)``.

    :param str uri: The URI which needs to be parsed into a reference.
    :param str encoding: The encoding of the string provided
    :returns: A parsed URI
    :rtype: :class:`URIReference`
    """
    return URIReference.from_string(uri, encoding)