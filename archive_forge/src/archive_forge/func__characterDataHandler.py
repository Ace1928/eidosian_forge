from fontTools import ttLib
from fontTools.misc.textTools import safeEval
from fontTools.ttLib.tables.DefaultTable import DefaultTable
import sys
import os
import logging
def _characterDataHandler(self, data):
    if self.stackSize > 1:
        if data != '\n' and self.contentStack[-1] and isinstance(self.contentStack[-1][-1], str) and (self.contentStack[-1][-1] != '\n'):
            self.contentStack[-1][-1] += data
        else:
            self.contentStack[-1].append(data)