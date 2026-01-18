import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def treeStateChanged(self, param, changes):
    """
        Called when the state of any sub-parameter has changed. 
        
        ==============  ================================================================
        **Arguments:**
        param           The immediate child whose tree state has changed.
                        note that the change may have originated from a grandchild.
        changes         List of tuples describing all changes that have been made
                        in this event: (param, changeDescr, data)
        ==============  ================================================================
                     
        This function can be extended to react to tree state changes.
        """
    self.treeStateChanges.extend(changes)
    self.emitTreeChanges()