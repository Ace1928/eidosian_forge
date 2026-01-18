import re
from contextlib import ExitStack
from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.PenPreviewLabel import PenPreviewLabel
from . import GroupParameterItem, WidgetParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter
@staticmethod
def updateFromPen(param, pen):
    """
        Applies settings from a pen to either a Parameter or dict. The Parameter or dict must already
        be populated with the relevant keys that can be found in `PenSelectorDialog.mkParam`.
        """
    stack = ExitStack()
    if isinstance(param, Parameter):
        names = param.names
        stack.enter_context(param.treeChangeBlocker())
    else:
        names = param
    for opt in names:
        if isinstance(param[opt], bool):
            attrName = f'is{opt.title()}'
        else:
            attrName = opt
        param[opt] = getattr(pen, attrName)()
    stack.close()