class NeverIndexedHeaderTuple(HeaderTuple):
    """
    A data structure that stores a single header field that cannot be added to
    a HTTP/2 header compression context.
    """
    __slots__ = ()
    indexable = False