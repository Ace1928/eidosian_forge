import sys
import re
import copy
import time
import os.path
def CPP_INTEGER(t):
    """(((((0x)|(0X))[0-9a-fA-F]+)|(\\d+))([uU][lL]|[lL][uU]|[uU]|[lL])?)"""
    return t