import re
from contextlib import ExitStack
from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.PenPreviewLabel import PenPreviewLabel
from . import GroupParameterItem, WidgetParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter

        Applies settings from a pen to either a Parameter or dict. The Parameter or dict must already
        be populated with the relevant keys that can be found in `PenSelectorDialog.mkParam`.
        