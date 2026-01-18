import keyword
import os
import pkgutil
import re
import subprocess
import sys
from argparse import Namespace
from collections import OrderedDict
from functools import lru_cache
import pyqtgraph as pg
from pyqtgraph.Qt import QT_LIB, QtCore, QtGui, QtWidgets
import exampleLoaderTemplate_generic as ui_template
import utils
def updateCodeViewTabWidth(self, font):
    """
        Change the codeView tabStopDistance to 4 spaces based on the size of the current font
        """
    fm = QtGui.QFontMetrics(font)
    tabWidth = fm.horizontalAdvance(' ' * 4)
    self.ui.codeView.setTabStopDistance(tabWidth)