from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _valuesview_repr(view):
    lst = []
    for v in view:
        lst.append('{!r}'.format(v))
    body = ', '.join(lst)
    return '{}({})'.format(view.__class__.__name__, body)