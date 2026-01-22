import datetime
import os
import re
import sys
from collections import OrderedDict
import numpy
from . import units
from .colormap import ColorMap
from .Point import Point
from .Qt import QtCore

configfile.py - Human-readable text configuration file library 
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.

Used for reading and writing dictionary objects to a python-like configuration
file format. Data structures may be nested and contain any data type as long
as it can be converted to/from a string using repr and eval.
