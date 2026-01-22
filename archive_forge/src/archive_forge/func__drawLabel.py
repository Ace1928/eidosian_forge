import copy
import functools
import math
import numpy
from rdkit import Chem
def _drawLabel(self, label, pos, baseOffset, font, color=None, **kwargs):
    color = color or self.drawingOptions.defaultColor
    x1 = pos[0]
    y1 = pos[1]
    labelSize = self.canvas.addCanvasText(label, (x1, y1, baseOffset), font, color, **kwargs)
    return labelSize