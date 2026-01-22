import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def __add_reaction_entries(self):
    """Add Entry elements for Reactions to the map drawing (PRIVATE).

        In KGML, these are typically line objects, so we render them
        before the compound circles to cover the unsightly ends/junctions
        """
    for reaction in self.pathway.reaction_entries:
        for g in reaction.graphics:
            self.drawing.setStrokeColor(color_to_reportlab(g.fgcolor))
            self.drawing.setFillColor(color_to_reportlab(g.bgcolor))
            self.__add_graphics(g)
            if self.label_reaction_entries:
                self.drawing.setFillColor(darken(g.fgcolor))
                self.__add_labels(g)