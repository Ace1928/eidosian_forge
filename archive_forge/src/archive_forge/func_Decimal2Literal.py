import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def Decimal2Literal(o, d):
    return format(o, 'f')