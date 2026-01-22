from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
def _isRibbi(nametable, nameID):
    englishRecord = nametable.getName(nameID, 3, 1, 1033)
    return True if englishRecord is not None and englishRecord.toUnicode() in ('Regular', 'Italic', 'Bold', 'Bold Italic') else False