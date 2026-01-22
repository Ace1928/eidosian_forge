import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def __add_compounds(self):
    """Add compound elements to the drawing of the map (PRIVATE)."""
    for compound in self.pathway.compounds:
        for g in compound.graphics:
            fillcolor = color_to_reportlab(g.bgcolor)
            if not compound.is_reactant:
                fillcolor.alpha *= self.non_reactant_transparency
            self.drawing.setStrokeColor(color_to_reportlab(g.fgcolor))
            self.drawing.setFillColor(fillcolor)
            self.__add_graphics(g)
            if self.label_compounds:
                if not compound.is_reactant:
                    t = 0.3
                else:
                    t = 1
                self.drawing.setFillColor(colors.Color(0.2, 0.2, 0.2, t))
                self.__add_labels(g)