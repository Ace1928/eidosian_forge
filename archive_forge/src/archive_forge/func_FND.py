import re, time, datetime
from .utils import isStr
def FND(d):
    """convert to ND if required"""
    return isinstance(d, NormalDate) and d or ND(d)