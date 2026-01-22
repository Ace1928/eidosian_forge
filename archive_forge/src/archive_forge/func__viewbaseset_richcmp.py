from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _viewbaseset_richcmp(view, other, op):
    if op == 0:
        if not isinstance(other, Set):
            return NotImplemented
        return len(view) < len(other) and view <= other
    elif op == 1:
        if not isinstance(other, Set):
            return NotImplemented
        if len(view) > len(other):
            return False
        for elem in view:
            if elem not in other:
                return False
        return True
    elif op == 2:
        if not isinstance(other, Set):
            return NotImplemented
        return len(view) == len(other) and view <= other
    elif op == 3:
        return not view == other
    elif op == 4:
        if not isinstance(other, Set):
            return NotImplemented
        return len(view) > len(other) and view >= other
    elif op == 5:
        if not isinstance(other, Set):
            return NotImplemented
        if len(view) < len(other):
            return False
        for elem in other:
            if elem not in view:
                return False
        return True