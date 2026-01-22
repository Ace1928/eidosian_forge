from fontTools import ttLib
from fontTools.misc.textTools import safeEval
from fontTools.ttLib.tables.DefaultTable import DefaultTable
import sys
import os
import logging
def _endElementHandler(self, name):
    self.stackSize = self.stackSize - 1
    del self.contentStack[-1]
    if not self.contentOnly:
        if self.stackSize == 1:
            self.root = None
        elif self.stackSize == 2:
            name, attrs, content = self.root
            self.currentTable.fromXML(name, attrs, content, self.ttFont)
            self.root = None