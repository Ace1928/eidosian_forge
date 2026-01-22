import array
import math
import os
import re
from PIL import Image
from rdkit.Chem.Draw.canvasbase import CanvasBase
def _doLine(self, p1, p2, **kwargs):
    if kwargs.get('dash', (0, 0)) == (0, 0):
        self.ctx.move_to(p1[0], p1[1])
        self.ctx.line_to(p2[0], p2[1])
    else:
        dash = kwargs['dash']
        pts = self._getLinePoints(p1, p2, dash)
        currDash = 0
        dashOn = True
        while currDash < len(pts) - 1:
            if dashOn:
                p1 = pts[currDash]
                p2 = pts[currDash + 1]
                self.ctx.move_to(p1[0], p1[1])
                self.ctx.line_to(p2[0], p2[1])
            currDash += 1
            dashOn = not dashOn