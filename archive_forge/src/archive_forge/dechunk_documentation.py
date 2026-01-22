from __future__ import annotations
from typing import TYPE_CHECKING, cast
from contourpy._contourpy import FillType, LineType
from contourpy.array import (
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.typecheck import check_filled, check_lines
Return the specified contour lines with all chunked data moved into the first chunk.

    Contour lines that are not chunked (``LineType.Separate`` and ``LineType.SeparateCode``) and
    those that are but only contain a single chunk are returned unmodified. Individual lines are
    unchanged, they are not geometrically combined.

    Args:
        lines (sequence of arrays): Contour line data as returned by
            :func:`~contourpy.ContourGenerator.lines`.
        line_type (LineType or str): Type of ``lines`` as enum or string equivalent.

    Return:
        Contour lines in a single chunk.

    .. versionadded:: 1.2.0
    