import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def topIsLayout(self):
    return isinstance(self[-1], QtWidgets.QLayout)