import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def __add_graphics(self, graphics):
    """Add the passed graphics object to the map (PRIVATE).

        Add text, add after the graphics object, for sane Z-ordering.
        """
    if graphics.type == 'line':
        p = self.drawing.beginPath()
        x, y = graphics.coords[0]
        if graphics.width is not None:
            self.drawing.setLineWidth(graphics.width)
        else:
            self.drawing.setLineWidth(1)
        p.moveTo(x, y)
        for x, y in graphics.coords:
            p.lineTo(x, y)
        self.drawing.drawPath(p)
        self.drawing.setLineWidth(1)
    if graphics.type == 'circle':
        self.drawing.circle(graphics.x, graphics.y, graphics.width * 0.5, stroke=1, fill=1)
    elif graphics.type == 'roundrectangle':
        self.drawing.roundRect(graphics.x - graphics.width * 0.5, graphics.y - graphics.height * 0.5, graphics.width, graphics.height, min(graphics.width, graphics.height) * 0.1, stroke=1, fill=1)
    elif graphics.type == 'rectangle':
        self.drawing.rect(graphics.x - graphics.width * 0.5, graphics.y - graphics.height * 0.5, graphics.width, graphics.height, stroke=1, fill=1)