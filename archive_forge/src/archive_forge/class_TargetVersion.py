from dataclasses import dataclass, field
from enum import Enum, auto
from hashlib import sha256
from operator import attrgetter
from typing import Dict, Final, Set
from black.const import DEFAULT_LINE_LENGTH
class TargetVersion(Enum):
    PY33 = 3
    PY34 = 4
    PY35 = 5
    PY36 = 6
    PY37 = 7
    PY38 = 8
    PY39 = 9
    PY310 = 10
    PY311 = 11
    PY312 = 12