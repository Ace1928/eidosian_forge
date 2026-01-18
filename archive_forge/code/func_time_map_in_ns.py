from collections import defaultdict
from typing import List, Dict, Any, Tuple, Iterator, Optional, Union
import numpy as np
from qiskit import pulse
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.pulse_v2.device_info import DrawerBackendInfo
def time_map_in_ns(time_window: Tuple[int, int], axis_breaks: List[Tuple[int, int]], dt: Optional[float]=None) -> types.HorizontalAxis:
    """Layout function for the horizontal axis formatting.

    Calculate axis break and map true time to axis labels. Generate equispaced
    6 horizontal axis ticks. Convert into seconds if ``dt`` is provided.

    Args:
        time_window: Left and right edge of this graph.
        axis_breaks: List of axis break period.
        dt: Time resolution of system.

    Returns:
        Axis formatter object.
    """
    t0, t1 = time_window
    t0_shift = t0
    t1_shift = t1
    axis_break_pos = []
    offset_accumulation = 0
    for t0b, t1b in axis_breaks:
        if t1b < t0 or t0b > t1:
            continue
        if t0 > t1b:
            t0_shift -= t1b - t0b
        if t1 > t1b:
            t1_shift -= t1b - t0b
        axis_break_pos.append(t0b - offset_accumulation)
        offset_accumulation += t1b - t0b
    axis_loc = np.linspace(max(t0_shift, 0), t1_shift, 6)
    axis_label = axis_loc.copy()
    for t0b, t1b in axis_breaks:
        offset = t1b - t0b
        axis_label = np.where(axis_label > t0b, axis_label + offset, axis_label)
    if dt:
        label = 'Time (ns)'
        axis_label *= dt * 1000000000.0
    else:
        label = 'System cycle time (dt)'
    formatted_label = [f'{val:.0f}' for val in axis_label]
    return types.HorizontalAxis(window=(t0_shift, t1_shift), axis_map=dict(zip(axis_loc, formatted_label)), axis_break_pos=axis_break_pos, label=label)