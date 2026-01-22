import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
def _layout_mode_text(self, space_vertically: bool=True, scale_weight: float=1.25, strip_rotated: bool=True, debug_path: Optional[Path]=None) -> str:
    """
        Get text preserving fidelity to source PDF text layout.

        Args:
            space_vertically: include blank lines inferred from y distance + font
                height. Defaults to True.
            scale_weight: multiplier for string length when calculating weighted
                average character width. Defaults to 1.25.
            strip_rotated: Removes text that is rotated w.r.t. to the page from
                layout mode output. Defaults to True.
            debug_path (Path | None): if supplied, must target a directory.
                creates the following files with debug information for layout mode
                functions if supplied:
                  - fonts.json: output of self._layout_mode_fonts
                  - tjs.json: individual text render ops with corresponding transform matrices
                  - bts.json: text render ops left justified and grouped by BT/ET operators
                  - bt_groups.json: BT/ET operations grouped by rendered y-coord (aka lines)
                Defaults to None.

        Returns:
            str: multiline string containing page text in a fixed width format that
                closely adheres to the rendered layout in the source pdf.
        """
    fonts = self._layout_mode_fonts()
    if debug_path:
        import json
        debug_path.joinpath('fonts.json').write_text(json.dumps(fonts, indent=2, default=lambda x: getattr(x, 'to_dict', str)(x)), 'utf-8')
    ops = iter(ContentStream(self['/Contents'].get_object(), self.pdf, 'bytes').operations)
    bt_groups = _layout_mode.text_show_operations(ops, fonts, strip_rotated, debug_path)
    if not bt_groups:
        return ''
    ty_groups = _layout_mode.y_coordinate_groups(bt_groups, debug_path)
    char_width = _layout_mode.fixed_char_width(bt_groups, scale_weight)
    return _layout_mode.fixed_width_page(ty_groups, char_width, space_vertically)