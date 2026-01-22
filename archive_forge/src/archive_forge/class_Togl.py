import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
class Togl(Widget):
    """
    Togl Widget
    Keith Junius
    Department of Biophysical Chemistry
    University of Groningen, The Netherlands
    Very basic widget which provides access to Togl functions.
    """

    def __init__(self, master=None, cnf={}, **kw):
        Widget.__init__(self, master, 'togl', cnf, kw)

    def render(self):
        return

    def swapbuffers(self):
        self.tk.call(self._w, 'swapbuffers')

    def makecurrent(self):
        self.tk.call(self._w, 'makecurrent')

    def alloccolor(self, red, green, blue):
        return self.tk.getint(self.tk.call(self._w, 'alloccolor', red, green, blue))

    def freecolor(self, index):
        self.tk.call(self._w, 'freecolor', index)

    def setcolor(self, index, red, green, blue):
        self.tk.call(self._w, 'setcolor', index, red, green, blue)

    def loadbitmapfont(self, fontname):
        return self.tk.getint(self.tk.call(self._w, 'loadbitmapfont', fontname))

    def unloadbitmapfont(self, fontbase):
        self.tk.call(self._w, 'unloadbitmapfont', fontbase)

    def uselayer(self, layer):
        self.tk.call(self._w, 'uselayer', layer)

    def showoverlay(self):
        self.tk.call(self._w, 'showoverlay')

    def hideoverlay(self):
        self.tk.call(self._w, 'hideoverlay')

    def existsoverlay(self):
        return self.tk.getboolean(self.tk.call(self._w, 'existsoverlay'))

    def getoverlaytransparentvalue(self):
        return self.tk.getint(self.tk.call(self._w, 'getoverlaytransparentvalue'))

    def ismappedoverlay(self):
        return self.tk.getboolean(self.tk.call(self._w, 'ismappedoverlay'))

    def alloccoloroverlay(self, red, green, blue):
        return self.tk.getint(self.tk.call(self._w, 'alloccoloroverlay', red, green, blue))

    def freecoloroverlay(self, index):
        self.tk.call(self._w, 'freecoloroverlay', index)