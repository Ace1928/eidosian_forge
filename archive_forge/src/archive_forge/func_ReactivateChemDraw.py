import os
import tempfile
import time
import re
def ReactivateChemDraw(openDoc=True, showDoc=True):
    global cdApp, theDoc, theObjs
    cdApp.Visible = 1
    if openDoc:
        theDoc = cdApp.Documents.Add()
        if theDoc and showDoc:
            theDoc.Activate()
        theObjs = theDoc.Objects