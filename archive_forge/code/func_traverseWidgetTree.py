import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def traverseWidgetTree(self, elem):
    for child in iter(elem):
        try:
            handler = self.widgetTreeItemHandlers[child.tag]
        except KeyError:
            continue
        handler(self, child)