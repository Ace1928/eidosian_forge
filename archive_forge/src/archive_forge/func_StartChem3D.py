import os
import tempfile
import time
import re
def StartChem3D(visible=0):
    """ launches Chem3D """
    global c3dApp
    c3dApp = Dispatch('Chem3D.Application')
    if not c3dApp.Visible:
        c3dApp.Visible = visible