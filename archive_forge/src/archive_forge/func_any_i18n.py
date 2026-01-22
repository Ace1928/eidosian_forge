import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
@staticmethod
def any_i18n(*args):
    """ Return True if any argument appears to be an i18n string. """
    for a in args:
        if a is not None and (not isinstance(a, str)):
            return True
    return False