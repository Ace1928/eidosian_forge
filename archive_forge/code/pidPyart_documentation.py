import Fontmapping  # helps by mapping pid font classes to Pyart font names
import pyart
from rdkit.sping.PDF import pdfmetrics
from rdkit.sping.pid import *
Attempts to return proper font name.
        PDF uses a standard 14 fonts referred to
        by name. Default to self.defaultFont('Helvetica').
        The dictionary allows a layer of indirection to
        support a standard set of PIDDLE font names.