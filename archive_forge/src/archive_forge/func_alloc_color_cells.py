import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def alloc_color_cells(self, contiguous, colors, planes):
    return request.AllocColorCells(display=self.display, contiguous=contiguous, cmap=self.id, colors=colors, planes=planes)