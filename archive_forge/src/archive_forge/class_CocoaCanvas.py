from ctypes import *
from .base import Display, Screen, ScreenMode, Canvas
from pyglet.libs.darwin.cocoapy import CGDirectDisplayID, quartz, cf
from pyglet.libs.darwin.cocoapy import cfstring_to_string, cfarray_to_list
class CocoaCanvas(Canvas):

    def __init__(self, display, screen, nsview):
        super(CocoaCanvas, self).__init__(display)
        self.screen = screen
        self.nsview = nsview