import os
import tempfile
from win32com.client import Dispatch
from rdkit import Chem
 FIX: the surface display stuff here is all screwed up due to
    differences between the way PyMol and DSViewer handle surfaces.
    In PyMol they are essentially a display mode for the protein, so
    they don't need to be managed separately.
    In DSViewer, on the other hand, the surface is attached to the
    protein, but it needs to be hidden or shown on its own.  I haven't
    figured out how to do that yet.
    