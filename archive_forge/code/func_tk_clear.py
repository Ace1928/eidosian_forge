from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def tk_clear(self):
    for item in self.canvas_items:
        self.canvas.delete(item)