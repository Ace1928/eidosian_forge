import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def _listdirworks(self):
    try:
        self.path.listdir()
    except py.error.ENOENT:
        return False
    else:
        return True