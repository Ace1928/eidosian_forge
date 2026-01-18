import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def topIsLayoutWidget(self):
    if type(self[-1]) is not QtWidgets.QWidget:
        return False
    if len(self) < 2:
        return False
    parent = self[-2]
    return isinstance(parent, QtWidgets.QWidget) and type(parent) not in (QtWidgets.QMainWindow, QtWidgets.QStackedWidget, QtWidgets.QToolBox, QtWidgets.QTabWidget, QtWidgets.QScrollArea, QtWidgets.QMdiArea, QtWidgets.QWizard, QtWidgets.QDockWidget)