from __future__ import annotations
from math import log, pi, sin, sqrt
from ._binary import o8
class GradientFile:
    gradient = None

    def getpalette(self, entries=256):
        palette = []
        ix = 0
        x0, x1, xm, rgb0, rgb1, segment = self.gradient[ix]
        for i in range(entries):
            x = i / (entries - 1)
            while x1 < x:
                ix += 1
                x0, x1, xm, rgb0, rgb1, segment = self.gradient[ix]
            w = x1 - x0
            if w < EPSILON:
                scale = segment(0.5, 0.5)
            else:
                scale = segment((xm - x0) / w, (x - x0) / w)
            r = o8(int(255 * ((rgb1[0] - rgb0[0]) * scale + rgb0[0]) + 0.5))
            g = o8(int(255 * ((rgb1[1] - rgb0[1]) * scale + rgb0[1]) + 0.5))
            b = o8(int(255 * ((rgb1[2] - rgb0[2]) * scale + rgb0[2]) + 0.5))
            a = o8(int(255 * ((rgb1[3] - rgb0[3]) * scale + rgb0[3]) + 0.5))
            palette.append(r + g + b + a)
        return (b''.join(palette), 'RGBA')