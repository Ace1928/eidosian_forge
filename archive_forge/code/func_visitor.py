from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def visitor(fobj, lptr, depth, includes):
    if depth > 0:
        loc = lptr.contents
        includes.append(FileInclusion(loc.file, File(fobj), loc, depth))