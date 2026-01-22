import warnings
import weakref
from time import perf_counter, perf_counter_ns
from .. import debug as debug
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets, isQObjectAlive
from .mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent
def addParentContextMenus(self, item, menu, event):
    """
        Can be called by any item in the scene to expand its context menu to include parent context menus.
        Parents may implement getContextMenus to add new menus / actions to the existing menu.
        getContextMenus must accept 1 argument (the event that generated the original menu) and
        return a single QMenu or a list of QMenus.
        
        The final menu will look like:
        
            |    Original Item 1
            |    Original Item 2
            |    ...
            |    Original Item N
            |    ------------------
            |    Parent Item 1
            |    Parent Item 2
            |    ...
            |    Grandparent Item 1
            |    ...
            
        
        ==============  ==================================================
        **Arguments:**
        item            The item that initially created the context menu 
                        (This is probably the item making the call to this function)
        menu            The context menu being shown by the item
        event           The original event that triggered the menu to appear.
        ==============  ==================================================
        """
    menusToAdd = []
    while item is not self:
        item = item.parentItem()
        if item is None:
            item = self
        if not hasattr(item, 'getContextMenus'):
            continue
        subMenus = item.getContextMenus(event) or []
        if isinstance(subMenus, list):
            menusToAdd.extend(subMenus)
        else:
            menusToAdd.append(subMenus)
    existingActions = menu.actions()
    actsToAdd = []
    for menuOrAct in menusToAdd:
        if isinstance(menuOrAct, QtWidgets.QMenu):
            menuOrAct = menuOrAct.menuAction()
        elif not isinstance(menuOrAct, QtGui.QAction):
            raise Exception(f'Cannot add object {menuOrAct} (type={type(menuOrAct)}) to QMenu.')
        if menuOrAct not in existingActions:
            actsToAdd.append(menuOrAct)
    if actsToAdd:
        menu.addSeparator()
    menu.addActions(actsToAdd)
    return menu