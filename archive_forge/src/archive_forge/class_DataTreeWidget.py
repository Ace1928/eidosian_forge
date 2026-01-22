import traceback
import types
from collections import OrderedDict
import numpy as np
from ..Qt import QtWidgets
from .TableWidget import TableWidget
class DataTreeWidget(QtWidgets.QTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays)
    """

    def __init__(self, parent=None, data=None):
        QtWidgets.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(3)
        self.setHeaderLabels(['key / index', 'type', 'value'])
        self.setAlternatingRowColors(True)

    def setData(self, data, hideRoot=False):
        """data should be a dictionary."""
        self.clear()
        self.widgets = []
        self.nodes = {}
        self.buildTree(data, self.invisibleRootItem(), hideRoot=hideRoot)
        self.expandToDepth(3)
        self.resizeColumnToContents(0)

    def buildTree(self, data, parent, name='', hideRoot=False, path=()):
        if hideRoot:
            node = parent
        else:
            node = QtWidgets.QTreeWidgetItem([name, '', ''])
            parent.addChild(node)
        self.nodes[path] = node
        typeStr, desc, childs, widget = self.parse(data)
        if len(desc) > 100:
            desc = desc[:97] + '...'
            if widget is None:
                widget = QtWidgets.QPlainTextEdit(str(data))
                widget.setMaximumHeight(200)
                widget.setReadOnly(True)
        node.setText(1, typeStr)
        node.setText(2, desc)
        if widget is not None:
            self.widgets.append(widget)
            subnode = QtWidgets.QTreeWidgetItem(['', '', ''])
            node.addChild(subnode)
            self.setItemWidget(subnode, 0, widget)
            subnode.setFirstColumnSpanned(True)
        for key, data in childs.items():
            self.buildTree(data, node, str(key), path=path + (key,))

    def parse(self, data):
        """
        Given any python object, return:
          * type
          * a short string representation
          * a dict of sub-objects to be parsed
          * optional widget to display as sub-node
        """
        typeStr = type(data).__name__
        if typeStr == 'instance':
            typeStr += ': ' + data.__class__.__name__
        widget = None
        desc = ''
        childs = {}
        if isinstance(data, dict):
            desc = 'length=%d' % len(data)
            if isinstance(data, OrderedDict):
                childs = data
            else:
                try:
                    childs = OrderedDict(sorted(data.items()))
                except TypeError:
                    childs = OrderedDict(data.items())
        elif isinstance(data, (list, tuple)):
            desc = 'length=%d' % len(data)
            childs = OrderedDict(enumerate(data))
        elif HAVE_METAARRAY and (hasattr(data, 'implements') and data.implements('MetaArray')):
            childs = OrderedDict([('data', data.view(np.ndarray)), ('meta', data.infoCopy())])
        elif isinstance(data, np.ndarray):
            desc = 'shape=%s dtype=%s' % (data.shape, data.dtype)
            table = TableWidget()
            table.setData(data)
            table.setMaximumHeight(200)
            widget = table
        elif isinstance(data, types.TracebackType):
            frames = list(map(str.strip, traceback.format_list(traceback.extract_tb(data))))
            widget = QtWidgets.QPlainTextEdit('\n'.join(frames))
            widget.setMaximumHeight(200)
            widget.setReadOnly(True)
        else:
            desc = str(data)
        return (typeStr, desc, childs, widget)