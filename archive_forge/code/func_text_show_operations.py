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
def text_show_operations(ops: Iterator[Tuple[List[Any], bytes]], fonts: Dict[str, Font], strip_rotated: bool=True, debug_path: Optional[Path]=None) -> List[BTGroup]:
    """
    Extract text from BT/ET operator pairs.

    Args:
        ops (Iterator[Tuple[List, bytes]]): iterator of operators in content stream
        fonts (Dict[str, Font]): font dictionary
        strip_rotated: Removes text if rotated w.r.t. to the page. Defaults to True.
        debug_path (Path, optional): Path to a directory for saving debug output.

    Returns:
        List[BTGroup]: list of dicts of text rendered by each BT operator
    """
    state_mgr = TextStateManager()
    debug = bool(debug_path)
    bt_groups: List[BTGroup] = []
    tj_debug: List[TextStateParams] = []
    try:
        warned_rotation = False
        while True:
            operands, op = next(ops)
            if op in (b'BT', b'q'):
                bts, tjs = recurs_to_target_op(ops, state_mgr, b'ET' if op == b'BT' else b'Q', fonts, strip_rotated)
                if not warned_rotation and any((tj.rotated for tj in tjs)):
                    warned_rotation = True
                    if strip_rotated:
                        logger_warning('Rotated text discovered. Output will be incomplete.', __name__)
                    else:
                        logger_warning('Rotated text discovered. Layout will be degraded.', __name__)
                bt_groups.extend(bts)
                if debug:
                    tj_debug.extend(tjs)
            else:
                state_mgr.set_state_param(op, operands)
    except StopIteration:
        pass
    min_x = min((x['tx'] for x in bt_groups), default=0.0)
    bt_groups = [dict(ogrp, tx=ogrp['tx'] - min_x, displaced_tx=ogrp['displaced_tx'] - min_x) for ogrp in sorted(bt_groups, key=lambda x: (x['ty'] * x['flip_sort'], -x['tx']), reverse=True)]
    if debug_path:
        import json
        debug_path.joinpath('bts.json').write_text(json.dumps(bt_groups, indent=2, default=str), 'utf-8')
        debug_path.joinpath('tjs.json').write_text(json.dumps(tj_debug, indent=2, default=lambda x: getattr(x, 'to_dict', str)(x)), 'utf-8')
    return bt_groups