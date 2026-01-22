import logging
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from . import curves_to_quadratic
from .errors import (
def _glyphs_to_quadratic(glyphs, max_err, reverse_direction, stats, all_quadratic=True):
    """Do the actual conversion of a set of compatible glyphs, after arguments
    have been set up.

    Return True if the glyphs were modified, else return False.
    """
    try:
        segments_by_location = zip(*[_get_segments(g) for g in glyphs])
    except UnequalZipLengthsError:
        raise IncompatibleSegmentNumberError(glyphs)
    if not any(segments_by_location):
        return False
    glyphs_modified = reverse_direction
    new_segments_by_location = []
    incompatible = {}
    for i, segments in enumerate(segments_by_location):
        tag = segments[0][0]
        if not all((s[0] == tag for s in segments[1:])):
            incompatible[i] = [s[0] for s in segments]
        elif tag == 'curve':
            new_segments = _segments_to_quadratic(segments, max_err, stats, all_quadratic)
            if all_quadratic or new_segments != segments:
                glyphs_modified = True
            segments = new_segments
        new_segments_by_location.append(segments)
    if glyphs_modified:
        new_segments_by_glyph = zip(*new_segments_by_location)
        for glyph, new_segments in zip(glyphs, new_segments_by_glyph):
            _set_segments(glyph, new_segments, reverse_direction)
    if incompatible:
        raise IncompatibleSegmentTypesError(glyphs, segments=incompatible)
    return glyphs_modified