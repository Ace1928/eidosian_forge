import collections.abc
import io
import itertools
import types
import typing
def _is_origin_subtype_args(left: 'NormalizedTypeArgs', right: 'NormalizedTypeArgs', forward_refs: typing.Optional[typing.Mapping[str, type]]) -> typing.Optional[bool]:
    if isinstance(left, frozenset):
        if not isinstance(right, frozenset):
            return False
        excluded = left - right
        if not excluded:
            return True
        return all((any((_is_normal_subtype(e, r, forward_refs) for r in right)) for e in excluded))
    if isinstance(left, collections.abc.Sequence) and (not isinstance(left, NormalizedType)):
        if not isinstance(right, collections.abc.Sequence) or isinstance(right, NormalizedType):
            return False
        if left and left[-1].origin is not Ellipsis and right and (right[-1].origin is Ellipsis):
            return all((_is_origin_subtype_args(l, right[0], forward_refs) for l in left))
        if len(left) != len(right):
            return False
        return all((l is not None and r is not None and _is_origin_subtype_args(l, r, forward_refs) for l, r in itertools.zip_longest(left, right)))
    assert isinstance(left, NormalizedType)
    assert isinstance(right, NormalizedType)
    return _is_normal_subtype(left, right, forward_refs)