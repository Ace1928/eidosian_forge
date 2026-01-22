import os
import tempfile
import time
import re
def StartChemDraw(visible=True, openDoc=False, showDoc=False):
    """ launches chemdraw """
    global cdApp, theDoc, theObjs, selectItem, cleanItem, centerItem
    if cdApp is not None:
        holder = None
        selectItem = None
        cleanItem = None
        centerItem = None
        theObjs = None
        theDoc = None
        cdApp = None
    cdApp = Dispatch('ChemDraw.Application')
    if openDoc:
        theDoc = cdApp.Documents.Add()
        theObjs = theDoc.Objects
    else:
        theDoc = None
    selectItem = cdApp.MenuBars(1).Menus(2).MenuItems(8)
    cleanItem = cdApp.MenuBars(1).Menus(5).MenuItems(6)
    if _cdxVersion == 6:
        centerItem = cdApp.MenuBars(1).Menus(4).MenuItems(1)
    else:
        centerItem = cdApp.MenuBars(1).Menus(4).MenuItems(7)
    if visible:
        cdApp.Visible = 1
        if theDoc and showDoc:
            theDoc.Activate()