import warnings
from ...Qt import QtCore
from .action import ParameterControlledButton
from .basetypes import GroupParameter, GroupParameterItem
from ..ParameterItem import ParameterItem
from ...Qt import QtCore, QtWidgets
def treeWidgetChanged(self):
    ParameterItem.treeWidgetChanged(self)
    tw = self.treeWidget()
    if tw is None:
        return
    tw.setItemWidget(self, 1, self.itemWidget)