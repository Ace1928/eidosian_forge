from __future__ import annotations
import functools
import typing
import warnings
def _adjust_focus_on_contents_modified(self, slc: slice, new_items: Collection[_T]=()) -> int:
    """
        Default behaviour is to move the focus to the item following
        any removed items, unless that item was simply replaced.

        Failing that choose the last item in the list.

        returns focus position for after change is applied
        """
    num_new_items = len(new_items)
    start, stop, step = indices = slc.indices(len(self))
    num_removed = len(list(range(*indices)))
    focus = self._validate_contents_modified(indices, new_items)
    if focus is not None:
        return focus
    focus = self._focus
    if step == 1:
        if start + num_new_items <= focus < stop:
            focus = stop
        if stop <= focus:
            focus += num_new_items - (stop - start)
    elif not num_new_items:
        if focus in range(start, stop, step):
            focus += 1
        focus -= len(list(range(start, min(focus, stop), step)))
    return min(focus, len(self) + num_new_items - num_removed - 1)