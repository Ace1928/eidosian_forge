from ._make import Attribute
def _split_what(what):
    """
    Returns a tuple of `frozenset`s of classes and attributes.
    """
    return (frozenset((cls for cls in what if isinstance(cls, type))), frozenset((cls for cls in what if isinstance(cls, str))), frozenset((cls for cls in what if isinstance(cls, Attribute))))