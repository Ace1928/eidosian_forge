import builtins
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def addClicked(self):
    """Called when "add new" button is clicked
        The parameter MUST have an 'addNew' method defined.
        """
    self.param.addNew()