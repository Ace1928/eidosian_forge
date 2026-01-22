from __future__ import annotations
from typing import TYPE_CHECKING
class LeafClassesMeta(type):
    """
    A metaclass for classes that keeps track of all of them that
    aren't base classes.

    >>> Parent = LeafClassesMeta('MyParentClass', (), {})
    >>> Parent in Parent._leaf_classes
    True
    >>> Child = LeafClassesMeta('MyChildClass', (Parent,), {})
    >>> Child in Parent._leaf_classes
    True
    >>> Parent in Parent._leaf_classes
    False

    >>> Other = LeafClassesMeta('OtherClass', (), {})
    >>> Parent in Other._leaf_classes
    False
    >>> len(Other._leaf_classes)
    1
    """
    _leaf_classes: set[type[Any]]

    def __init__(cls, name: str, bases: tuple[type[object], ...], attrs: dict[str, object]) -> None:
        if not hasattr(cls, '_leaf_classes'):
            cls._leaf_classes = set()
        leaf_classes = getattr(cls, '_leaf_classes')
        leaf_classes.add(cls)
        leaf_classes -= set(bases)