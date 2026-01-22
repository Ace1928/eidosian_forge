import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def __add_labels(self, graphics):
    """Add labels for the passed graphics objects to the map (PRIVATE).

        We don't check that the labels fit inside objects such as circles/
        rectangles/roundrectangles.
        """
    if graphics.type == 'line':
        mid_idx = len(graphics.coords) * 0.5
        if int(mid_idx) != mid_idx:
            idx1, idx2 = (int(mid_idx - 0.5), int(mid_idx + 0.5))
        else:
            idx1, idx2 = (int(mid_idx - 1), int(mid_idx))
        x1, y1 = graphics.coords[idx1]
        x2, y2 = graphics.coords[idx2]
        x, y = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
    elif graphics.type == 'circle':
        x, y = (graphics.x, graphics.y)
    elif graphics.type in ('rectangle', 'roundrectangle'):
        x, y = (graphics.x, graphics.y)
    if graphics._parent.type == 'map':
        text = graphics.name
        self.drawing.setFont(self.fontname, self.fontsize + 2)
    elif len(graphics.name) < 15:
        text = graphics.name
    else:
        text = graphics.name[:12] + '...'
    self.drawing.drawCentredString(x, y, text)
    self.drawing.setFont(self.fontname, self.fontsize)