from __future__ import annotations
from matplotlib.offsetbox import (
from matplotlib.patches import bbox_artist as mbbox_artist
from matplotlib.transforms import Affine2D, Bbox
from .patches import InsideStrokedRectangle
def _correct_dpi(self, renderer):
    if not self._dpi_corrected:
        dpi_cor = renderer.points_to_pixels(1.0)
        self.dpi_transform.clear()
        self.dpi_transform.scale(dpi_cor, dpi_cor)
        self._dpi_corrected = True