from __future__ import annotations
import functools
import typing
import warnings
class MonitoredList(typing.List[_T], typing.Generic[_T]):
    """
    This class can trigger a callback any time its contents are changed
    with the usual list operations append, extend, etc.
    """

    def _modified(self) -> None:
        pass

    def set_modified_callback(self, callback: Callable[[], typing.Any]) -> None:
        """
        Assign a callback function with no parameters that is called any
        time the list is modified.  Callback's return value is ignored.

        >>> import sys
        >>> ml = MonitoredList([1,2,3])
        >>> ml.set_modified_callback(lambda: sys.stdout.write("modified\\n"))
        >>> ml
        MonitoredList([1, 2, 3])
        >>> ml.append(10)
        modified
        >>> len(ml)
        4
        >>> ml += [11, 12, 13]
        modified
        >>> ml[:] = ml[:2] + ml[-2:]
        modified
        >>> ml
        MonitoredList([1, 2, 12, 13])
        """
        self._modified = callback

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({list(self)!r})'

    def __rich_repr__(self) -> Iterator[tuple[str | None, typing.Any] | typing.Any]:
        for item in self:
            yield (None, item)
    __add__ = _call_modified(list.__add__)
    __delitem__ = _call_modified(list.__delitem__)
    __iadd__ = _call_modified(list.__iadd__)
    __imul__ = _call_modified(list.__imul__)
    __rmul__ = _call_modified(list.__rmul__)
    __setitem__ = _call_modified(list.__setitem__)
    append = _call_modified(list.append)
    extend = _call_modified(list.extend)
    insert = _call_modified(list.insert)
    pop = _call_modified(list.pop)
    remove = _call_modified(list.remove)
    reverse = _call_modified(list.reverse)
    sort = _call_modified(list.sort)
    if hasattr(list, 'clear'):
        clear = _call_modified(list.clear)