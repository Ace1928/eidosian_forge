import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def __add_orthologs(self):
    """Add 'ortholog' Entry elements to the drawing of the map (PRIVATE).

        In KGML, these are typically line objects, so we render them
        before the compound circles to cover the unsightly ends/junctions.
        """
    for ortholog in self.pathway.orthologs:
        for g in ortholog.graphics:
            self.drawing.setStrokeColor(color_to_reportlab(g.fgcolor))
            self.drawing.setFillColor(color_to_reportlab(g.bgcolor))
            self.__add_graphics(g)
            if self.label_orthologs:
                self.drawing.setFillColor(darken(g.fgcolor))
                self.__add_labels(g)