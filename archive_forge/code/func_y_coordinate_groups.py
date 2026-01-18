import sys
from itertools import groupby
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from ..._utils import logger_warning
from .. import LAYOUT_NEW_BT_GROUP_SPACE_WIDTHS
from ._font import Font
from ._text_state_manager import TextStateManager
from ._text_state_params import TextStateParams
def y_coordinate_groups(bt_groups: List[BTGroup], debug_path: Optional[Path]=None) -> Dict[int, List[BTGroup]]:
    """
    Group text operations by rendered y coordinate, i.e. the line number.

    Args:
        bt_groups: list of dicts as returned by text_show_operations()
        debug_path (Path, optional): Path to a directory for saving debug output.

    Returns:
        Dict[int, List[BTGroup]]: dict of lists of text rendered by each BT operator
            keyed by y coordinate
    """
    ty_groups = {ty: sorted(grp, key=lambda x: x['tx']) for ty, grp in groupby(bt_groups, key=lambda bt_grp: int(bt_grp['ty'] * bt_grp['flip_sort']))}
    last_ty = next(iter(ty_groups))
    last_txs = {int(_t['tx']) for _t in ty_groups[last_ty] if _t['text'].strip()}
    for ty in list(ty_groups)[1:]:
        fsz = min((ty_groups[_y][0]['font_height'] for _y in (ty, last_ty)))
        txs = {int(_t['tx']) for _t in ty_groups[ty] if _t['text'].strip()}
        no_text_overlap = not txs & last_txs
        offset_less_than_font_height = abs(ty - last_ty) < fsz
        if no_text_overlap and offset_less_than_font_height:
            ty_groups[last_ty] = sorted(ty_groups.pop(ty) + ty_groups[last_ty], key=lambda x: x['tx'])
            last_txs |= txs
        else:
            last_ty = ty
            last_txs = txs
    if debug_path:
        import json
        debug_path.joinpath('bt_groups.json').write_text(json.dumps(ty_groups, indent=2, default=str), 'utf-8')
    return ty_groups