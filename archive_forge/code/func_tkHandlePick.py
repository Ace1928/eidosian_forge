import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkHandlePick(self, event):
    """Handle a pick on the scene."""
    if hasattr(self, 'pick'):
        realy = self.winfo_height() - event.y
        p1 = gluUnProject(event.x, realy, 0.0)
        p2 = gluUnProject(event.x, realy, 1.0)
        if self.pick(self, p1, p2):
            'If the pick method returns true we redraw the scene.'
            self.tkRedraw()