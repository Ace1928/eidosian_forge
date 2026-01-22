import os
import re
import weakref
from rdkit import Chem, RDConfig
class FuncGroupFileParseError(ValueError):
    pass