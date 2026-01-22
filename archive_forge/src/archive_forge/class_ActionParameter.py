from ...Qt import QtCore, QtWidgets, QtGui
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
class ActionParameter(Parameter):
    """
    Used for displaying a button within the tree.

    ``sigActivated(self)`` is emitted when the button is clicked.

    Parameters
    ----------
    icon: str
        Icon to display in the button. Can be any argument accepted
        by :class:`QIcon <QtGui.QIcon>`.
    shortcut: str
        Key sequence to use as a shortcut for the button. Note that this shortcut is
        associated with spawned parameters, i.e. the shortcut will only work when this
        parameter has an item in a tree that is visible. Can be set to any string
        accepted by :class:`QKeySequence <QtGui.QKeySequence>`.
    """
    itemClass = ActionParameterItem
    sigActivated = QtCore.Signal(object)

    def activate(self):
        self.sigActivated.emit(self)
        self.emitStateChanged('activated', None)