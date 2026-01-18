import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
center of physical surface is at 0,0
        radius is the radius of the surface. If radius is None, the surface is flat. 
        diameter is of the optic's edge.