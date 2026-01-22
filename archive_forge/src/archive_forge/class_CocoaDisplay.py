from ctypes import *
from .base import Display, Screen, ScreenMode, Canvas
from pyglet.libs.darwin.cocoapy import CGDirectDisplayID, quartz, cf
from pyglet.libs.darwin.cocoapy import cfstring_to_string, cfarray_to_list
class CocoaDisplay(Display):

    def get_screens(self):
        maxDisplays = 256
        activeDisplays = (CGDirectDisplayID * maxDisplays)()
        count = c_uint32()
        quartz.CGGetActiveDisplayList(maxDisplays, activeDisplays, byref(count))
        return [CocoaScreen(self, displayID) for displayID in list(activeDisplays)[:count.value]]