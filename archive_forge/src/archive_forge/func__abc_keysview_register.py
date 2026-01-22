from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _abc_keysview_register(view_cls):
    KeysView.register(view_cls)