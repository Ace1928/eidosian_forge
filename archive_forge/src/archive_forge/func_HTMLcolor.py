import tkinter
from tkinter.constants import *
def HTMLcolor(canvas, color):
    """returns Tk color in form '#rrggbb' or '#rgb'"""
    if color:
        r, g, b = ['%02x' % (c // 256) for c in canvas.winfo_rgb(color)]
        if r[0] == r[1] and g[0] == g[1] and (b[0] == b[1]):
            return '#' + r[0] + g[0] + b[0]
        else:
            return '#' + r + g + b
    else:
        return color