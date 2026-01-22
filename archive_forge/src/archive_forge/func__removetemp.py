import py
import os
import sys
import marshal
def _removetemp(self):
    if self.tempdir.check():
        self.tempdir.remove()