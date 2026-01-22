from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
def _fontVersion(font, platform=(3, 1, 1033)):
    nameRecord = font['name'].getName(NameID.VERSION_STRING, *platform)
    if nameRecord is None:
        return f'{font['head'].fontRevision:.3f}'
    versionNumber = nameRecord.toUnicode().split(';')[0]
    return versionNumber.lstrip('Version ').strip()