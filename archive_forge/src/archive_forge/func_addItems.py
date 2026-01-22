import sys
from collections import OrderedDict
from ..Qt import QtWidgets
@ignoreIndexChange
@blockIfUnchanged
def addItems(self, items):
    if isinstance(items, list) or isinstance(items, tuple):
        texts = items
        items = dict([(x, x) for x in items])
    elif isinstance(items, dict):
        texts = list(items.keys())
    else:
        raise TypeError('items argument must be list or dict or tuple (got %s).' % type(items))
    for t in texts:
        if t in self._items:
            raise Exception('ComboBox already has item named "%s".' % t)
    for k, v in items.items():
        self._items[k] = v
    QtWidgets.QComboBox.addItems(self, list(texts))
    self.itemsChanged()