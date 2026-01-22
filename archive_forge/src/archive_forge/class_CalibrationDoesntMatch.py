from copy import copy
from dataclasses import dataclass
from typing import Union, Dict, List, Any, Optional, no_type_check
from pyquil.quilatom import (
from pyquil.quilbase import (
class CalibrationDoesntMatch(CalibrationError):
    pass