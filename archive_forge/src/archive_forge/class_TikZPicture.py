from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
class TikZPicture:

    def __init__(self, canvas, raw_colors, width=282.0):
        self.string = ''
        ulx, uly, lrx, lry = canvas.bbox(Tk_.ALL)
        pt_scale = float(width) / (lrx - ulx)
        cm_scale = 0.0352777778 * pt_scale
        self.transform = lambda xy: (cm_scale * (-ulx + xy[0]), cm_scale * (lry - xy[1]))
        self.colors = dict()
        for i, hex_color in enumerate(raw_colors):
            self.colors[hex_color] = i
            rgb = [int(c, 16) / 255.0 for c in in_twos(hex_color[1:])]
            self.string += '\\definecolor{linkcolor%d}' % i + '{rgb}{%.2f, %.2f, %.2f}\n' % tuple(rgb)
        self.string += '\\begin{tikzpicture}[line width=%.1f, line cap=round, line join=round]\n' % (pt_scale * 4)
        self.curcolor = None

    def write(self, color, line):
        if color != self.curcolor:
            if self.curcolor is not None:
                self.string += '  \\end{scope}\n'
            self.string += '  \\begin{scope}[color=linkcolor%d]\n' % self.colors[color]
            self.curcolor = color
        self.string += line

    def save(self, file_name):
        file = open(file_name, 'w')
        file.write(self.string + '  \\end{scope}\n\\end{tikzpicture}\n')
        file.close()