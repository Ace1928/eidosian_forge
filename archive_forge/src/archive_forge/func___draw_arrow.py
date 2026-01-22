import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def __draw_arrow(self, g_from, g_to):
    """Draw an arrow between given Entry objects (PRIVATE).

        Draws an arrow from the g_from Entry object to the g_to
        Entry object; both must have Graphics objects.
        """
    bounds_from, bounds_to = (g_from.bounds, g_to.bounds)
    centre_from = (0.5 * (bounds_from[0][0] + bounds_from[1][0]), 0.5 * (bounds_from[0][1] + bounds_from[1][1]))
    centre_to = (0.5 * (bounds_to[0][0] + bounds_to[1][0]), 0.5 * (bounds_to[0][1] + bounds_to[1][1]))
    p = self.drawing.beginPath()
    if bounds_to[0][0] < centre_from[0] < bounds_to[1][0]:
        if centre_to[1] > centre_from[1]:
            p.moveTo(centre_from[0], bounds_from[1][1])
            p.lineTo(centre_from[0], bounds_to[0][1])
        else:
            p.moveTo(centre_from[0], bounds_from[0][1])
            p.lineTo(centre_from[0], bounds_to[1][1])
    elif bounds_from[0][0] < centre_to[0] < bounds_from[1][0]:
        if centre_to[1] > centre_from[1]:
            p.moveTo(centre_to[0], bounds_from[1][1])
            p.lineTo(centre_to[0], bounds_to[0][1])
        else:
            p.moveTo(centre_to[0], bounds_from[0][1])
            p.lineTo(centre_to[0], bounds_to[1][1])
    self.drawing.drawPath(p)