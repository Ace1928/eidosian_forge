import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkAutoSpin(self, event):
    """Perform autospin of scene."""
    self.after(4)
    self.update_idletasks()
    x = self.tk.getint(self.tk.call('winfo', 'pointerx', self._w))
    y = self.tk.getint(self.tk.call('winfo', 'pointery', self._w))
    if self.autospin_allowed:
        if x != event.x_root and y != event.y_root:
            self.autospin = 1
    self.yspin = x - event.x_root
    self.xspin = y - event.y_root
    self.after(10, self.do_AutoSpin)