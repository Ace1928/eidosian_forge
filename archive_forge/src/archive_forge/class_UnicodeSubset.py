from sys import maxunicode
from typing import cast, Iterable, Iterator, List, MutableSet, Union, Optional
from .unicode_categories import RAW_UNICODE_CATEGORIES
from .codepoints import CodePoint, code_point_order, code_point_repr, \
class UnicodeSubset(MutableSet[CodePoint]):
    """
    Represents a subset of Unicode code points, implemented with an ordered list of
    integer values and ranges. Codepoints can be added or discarded using sequences
    of integer values and ranges or with strings equivalent to regex character set.

    :param codepoints: a sequence of integer values and ranges, another UnicodeSubset     instance ora a string equivalent of a regex character set.
    """
    __slots__ = ('_codepoints',)
    _codepoints: List[CodePoint]

    def __init__(self, codepoints: Optional[CodePointsArgType]=None) -> None:
        if not codepoints:
            self._codepoints = list()
        elif isinstance(codepoints, list):
            self._codepoints = sorted(codepoints, key=code_point_order)
        elif isinstance(codepoints, UnicodeSubset):
            self._codepoints = codepoints.codepoints.copy()
        else:
            self._codepoints = list()
            self.update(codepoints)

    @property
    def codepoints(self) -> List[CodePoint]:
        return self._codepoints

    def __repr__(self) -> str:
        return '%s(%r)' % (self.__class__.__name__, str(self))

    def __str__(self) -> str:
        return ''.join((code_point_repr(cp) for cp in self._codepoints))

    def copy(self) -> 'UnicodeSubset':
        return self.__copy__()

    def __copy__(self) -> 'UnicodeSubset':
        return UnicodeSubset(self._codepoints)

    def __reversed__(self) -> Iterator[int]:
        for item in reversed(self._codepoints):
            if isinstance(item, int):
                yield item
            else:
                yield from reversed(range(item[0], item[1]))

    def complement(self) -> Iterator[CodePoint]:
        last_cp = 0
        for cp in self._codepoints:
            if isinstance(cp, int):
                cp = (cp, cp + 1)
            diff = cp[0] - last_cp
            if diff > 2:
                yield (last_cp, cp[0])
            elif diff == 2:
                yield last_cp
                yield (last_cp + 1)
            elif diff == 1:
                yield last_cp
            elif diff:
                raise ValueError('unordered code points found in {!r}'.format(self))
            last_cp = cp[1]
        if last_cp < maxunicode:
            yield (last_cp, maxunicode + 1)
        elif last_cp == maxunicode:
            yield maxunicode

    def iter_characters(self) -> Iterator[str]:
        return map(chr, self.__iter__())

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int):
            try:
                value = ord(value)
            except TypeError:
                return False
        for cp in self._codepoints:
            if not isinstance(cp, int):
                if cp[0] > value:
                    return False
                elif cp[1] <= value:
                    continue
                else:
                    return True
            elif cp > value:
                return False
            elif cp == value:
                return True
        return False

    def __iter__(self) -> Iterator[int]:
        for cp in self._codepoints:
            if isinstance(cp, int):
                yield cp
            else:
                yield from range(*cp)

    def __len__(self) -> int:
        k = 0
        for _ in self:
            k += 1
        return k

    def update(self, *others: Union[str, Iterable[CodePoint]]) -> None:
        for value in others:
            if isinstance(value, str):
                for cp in iter_code_points(iterparse_character_subset(value), reverse=True):
                    self.add(cp)
            else:
                for cp in iter_code_points(value, reverse=True):
                    self.add(cp)

    def add(self, value: CodePoint) -> None:
        try:
            start_value, end_value = get_code_point_range(value)
        except TypeError:
            raise ValueError('{!r} is not a Unicode code point value/range'.format(value))
        code_points = self._codepoints
        last_index = len(code_points) - 1
        for k, cp in enumerate(code_points):
            if isinstance(cp, int):
                cp = (cp, cp + 1)
            if end_value < cp[0]:
                code_points.insert(k, value)
            elif start_value > cp[1]:
                continue
            elif end_value > cp[1]:
                if k == last_index:
                    code_points[k] = (min(cp[0], start_value), end_value)
                else:
                    next_cp = code_points[k + 1]
                    higher_bound = next_cp if isinstance(next_cp, int) else next_cp[0]
                    if end_value <= higher_bound:
                        code_points[k] = (min(cp[0], start_value), end_value)
                    else:
                        code_points[k] = (min(cp[0], start_value), higher_bound)
                        start_value = higher_bound
                        continue
            elif start_value < cp[0]:
                code_points[k] = (start_value, cp[1])
            break
        else:
            self._codepoints.append(value)

    def difference_update(self, *others: Union[str, Iterable[CodePoint]]) -> None:
        for value in others:
            if isinstance(value, str):
                for cp in iter_code_points(iterparse_character_subset(value), reverse=True):
                    self.discard(cp)
            else:
                for cp in iter_code_points(value, reverse=True):
                    self.discard(cp)

    def discard(self, value: CodePoint) -> None:
        try:
            start_cp, end_cp = get_code_point_range(value)
        except TypeError:
            raise ValueError('{!r} is not a Unicode code point value/range'.format(value))
        code_points = self._codepoints
        for k in reversed(range(len(code_points))):
            cp = code_points[k]
            if isinstance(cp, int):
                cp = (cp, cp + 1)
            if start_cp >= cp[1]:
                break
            elif end_cp >= cp[1]:
                if start_cp <= cp[0]:
                    del code_points[k]
                elif start_cp - cp[0] > 1:
                    code_points[k] = (cp[0], start_cp)
                else:
                    code_points[k] = cp[0]
            elif end_cp > cp[0]:
                if start_cp <= cp[0]:
                    if cp[1] - end_cp > 1:
                        code_points[k] = (end_cp, cp[1])
                    else:
                        code_points[k] = cp[1] - 1
                else:
                    if cp[1] - end_cp > 1:
                        code_points.insert(k + 1, (end_cp, cp[1]))
                    else:
                        code_points.insert(k + 1, cp[1] - 1)
                    if start_cp - cp[0] > 1:
                        code_points[k] = (cp[0], start_cp)
                    else:
                        code_points[k] = cp[0]

    def clear(self) -> None:
        del self._codepoints[:]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Iterable):
            return NotImplemented
        elif isinstance(other, UnicodeSubset):
            return self._codepoints == other._codepoints
        else:
            return self._codepoints == other

    def __ior__(self, other: object) -> 'UnicodeSubset':
        if not isinstance(other, Iterable):
            return NotImplemented
        elif isinstance(other, UnicodeSubset):
            other = reversed(other._codepoints)
        elif isinstance(other, str):
            other = reversed(UnicodeSubset(other)._codepoints)
        else:
            other = iter_code_points(other, reverse=True)
        for cp in other:
            self.add(cp)
        return self

    def __or__(self, other: object) -> 'UnicodeSubset':
        obj = self.copy()
        return obj.__ior__(other)

    def __isub__(self, other: object) -> 'UnicodeSubset':
        if not isinstance(other, Iterable):
            return NotImplemented
        elif isinstance(other, UnicodeSubset):
            other = reversed(other._codepoints)
        elif isinstance(other, str):
            other = reversed(UnicodeSubset(other)._codepoints)
        else:
            other = iter_code_points(other, reverse=True)
        for cp in other:
            self.discard(cp)
        return self

    def __sub__(self, other: object) -> 'UnicodeSubset':
        obj = self.copy()
        return obj.__isub__(other)
    __rsub__ = __sub__

    def __iand__(self, other: object) -> 'UnicodeSubset':
        if not isinstance(other, Iterable):
            return NotImplemented
        for value in self - other:
            self.discard(value)
        return self

    def __and__(self, other: object) -> 'UnicodeSubset':
        obj = self.copy()
        return obj.__iand__(other)

    def __ixor__(self, other: object) -> 'UnicodeSubset':
        if other is self:
            self.clear()
            return self
        elif not isinstance(other, Iterable):
            return NotImplemented
        elif not isinstance(other, UnicodeSubset):
            other = UnicodeSubset(cast(Union[str, Iterable[CodePoint]], other))
        for value in other:
            if value in self:
                self.discard(value)
            else:
                self.add(value)
        return self

    def __xor__(self, other: object) -> 'UnicodeSubset':
        obj = self.copy()
        return obj.__ixor__(other)