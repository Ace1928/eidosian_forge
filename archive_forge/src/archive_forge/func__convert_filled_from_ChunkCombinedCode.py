from __future__ import annotations
from typing import TYPE_CHECKING, cast
import numpy as np
from contourpy._contourpy import FillType, LineType
import contourpy.array as arr
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.typecheck import check_filled, check_lines
from contourpy.types import MOVETO, offset_dtype
def _convert_filled_from_ChunkCombinedCode(filled: cpy.FillReturn_ChunkCombinedCode, fill_type_to: FillType) -> cpy.FillReturn:
    if fill_type_to == FillType.ChunkCombinedCode:
        return filled
    elif fill_type_to == FillType.ChunkCombinedOffset:
        codes = [None if codes is None else arr.offsets_from_codes(codes) for codes in filled[1]]
        return (filled[0], codes)
    else:
        raise ValueError(f'Conversion from {FillType.ChunkCombinedCode} to {fill_type_to} not supported')