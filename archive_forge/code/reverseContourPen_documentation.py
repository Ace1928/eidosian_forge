from fontTools.misc.arrayTools import pairwise
from fontTools.pens.filterPen import ContourFilterPen
Generator that takes a list of pen's (operator, operands) tuples,
    and yields them with the winding direction reversed.
    