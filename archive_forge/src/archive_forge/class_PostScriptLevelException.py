import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
class PostScriptLevelException(ValueError):
    pass