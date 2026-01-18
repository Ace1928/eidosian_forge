import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def v3distsq(a, b):
    d = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]