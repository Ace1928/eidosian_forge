import sys
import re
import os
from configparser import RawConfigParser
def _escape_backslash(val):
    return val.replace('\\', '\\\\')