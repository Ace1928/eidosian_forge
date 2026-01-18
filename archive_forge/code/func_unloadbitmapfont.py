import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def unloadbitmapfont(self, fontbase):
    self.tk.call(self._w, 'unloadbitmapfont', fontbase)