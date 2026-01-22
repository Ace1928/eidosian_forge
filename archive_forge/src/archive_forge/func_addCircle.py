import array
import math
import os
import re
from PIL import Image
from rdkit.Chem.Draw.canvasbase import CanvasBase
def addCircle(self, center, radius, color=(0, 0, 0), fill=True, stroke=False, alpha=1.0, **kwargs):
    if not fill and (not stroke):
        return
    self.ctx.set_source_rgba(color[0], color[1], color[2], alpha)
    self.ctx.arc(center[0], center[1], radius, 0, 2.0 * math.pi)
    self.ctx.close_path()
    if stroke:
        if fill:
            self.ctx.stroke_preserve()
        else:
            self.ctx.stroke()
    if fill:
        self.ctx.fill()