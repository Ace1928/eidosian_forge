import collections.abc
import io
import itertools
import types
import typing
def _is_origin_subtype(left: OriginType, right: OriginType) -> bool:
    if left is right:
        return True
    if left is not None and left in STATIC_SUBTYPE_MAPPING and (right == STATIC_SUBTYPE_MAPPING[left]):
        return True
    if hasattr(left, 'mro'):
        for parent in left.mro():
            if parent == right:
                return True
    if isinstance(left, type) and isinstance(right, type):
        return issubclass(left, right)
    return left == right