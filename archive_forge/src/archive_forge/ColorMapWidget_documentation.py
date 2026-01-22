from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore

        Return an array of colors corresponding to *data*. 
        
        ==============  =================================================================
        **Arguments:**
        data            A numpy record array where the fields in data.dtype match those
                        defined by a prior call to setFields().
        mode            Either 'byte' or 'float'. For 'byte', the method returns an array
                        of dtype ubyte with values scaled 0-255. For 'float', colors are
                        returned as 0.0-1.0 float values.
        ==============  =================================================================
        