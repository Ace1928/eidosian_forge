from reportlab.lib import colors
from Bio.Graphics.BasicChromosome import ChromosomeSegment
from Bio.Graphics.BasicChromosome import TelomereSegment
def _color_from_count(self, count):
    """Translate the given count into a color using the color scheme (PRIVATE)."""
    for count_start, count_end in self._color_scheme:
        if count >= count_start and count <= count_end:
            return self._color_scheme[count_start, count_end]
    raise ValueError(f'Count value {count} was not found in the color scheme.')