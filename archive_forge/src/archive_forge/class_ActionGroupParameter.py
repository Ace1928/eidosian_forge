import warnings
from ...Qt import QtCore
from .action import ParameterControlledButton
from .basetypes import GroupParameter, GroupParameterItem
from ..ParameterItem import ParameterItem
from ...Qt import QtCore, QtWidgets
class ActionGroupParameter(GroupParameter):
    itemClass = ActionGroupParameterItem
    sigActivated = QtCore.Signal(object)

    def __init__(self, **opts):
        opts.setdefault('button', {})
        super().__init__(**opts)

    def activate(self):
        self.sigActivated.emit(self)
        self.emitStateChanged('activated', None)

    def setButtonOpts(self, **opts):
        """
        Update individual button options without replacing the entire
        button definition.
        """
        buttonOpts = self.opts.get('button', {}).copy()
        buttonOpts.update(opts)
        self.setOpts(button=buttonOpts)