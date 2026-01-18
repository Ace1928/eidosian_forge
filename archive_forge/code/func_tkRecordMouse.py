import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkRecordMouse(self, event):
    """Record the current mouse position."""
    self.xmouse = event.x
    self.ymouse = event.y