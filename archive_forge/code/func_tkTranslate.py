import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkTranslate(self, event):
    """Perform translation of scene."""
    self.activate()
    win_height = max(1, self.winfo_height())
    obj_c = (self.xcenter, self.ycenter, self.zcenter)
    win = gluProject(obj_c[0], obj_c[1], obj_c[2])
    obj = gluUnProject(win[0], win[1] + 0.5 * win_height, win[2])
    dist = math.sqrt(v3distsq(obj, obj_c))
    scale = abs(dist / (0.5 * win_height))
    glTranslateScene(scale, event.x, event.y, self.xmouse, self.ymouse)
    self.tkRedraw()
    self.tkRecordMouse(event)