from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _abc_itemsview_register(view_cls):
    ItemsView.register(view_cls)