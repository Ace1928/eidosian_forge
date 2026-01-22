from __future__ import annotations
from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
from fontTools.pens.recordingPen import RecordingPen
class DecomposingFilterPen(_DecomposingFilterPenMixin, DecomposingPen, FilterPen):
    """Filter pen that draws components as regular contours."""
    pass