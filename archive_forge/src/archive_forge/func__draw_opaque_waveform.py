from __future__ import annotations
import re
from fractions import Fraction
from typing import Any
import numpy as np
from qiskit import pulse, circuit
from qiskit.pulse import instructions, library
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def _draw_opaque_waveform(init_time: int, duration: int, pulse_shape: str, pnames: list[str], meta: dict[str, Any], channel: pulse.channels.PulseChannel, formatter: dict[str, Any]) -> list[drawings.LineData | drawings.BoxData | drawings.TextData]:
    """A private function that generates drawings of stepwise pulse lines.

    Args:
        init_time: Time when the opaque waveform starts.
        duration: Duration of opaque waveform. This can be None or ParameterExpression.
        pulse_shape: String that represents pulse shape.
        pnames: List of parameter names.
        meta: Metadata dictionary of the waveform.
        channel: Channel associated with the waveform to draw.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of drawings.
    """
    fill_objs: list[drawings.LineData | drawings.BoxData | drawings.TextData] = []
    fc, ec = formatter['color.opaque_shape']
    box_style = {'zorder': formatter['layer.fill_waveform'], 'alpha': formatter['alpha.opaque_shape'], 'linewidth': formatter['line_width.opaque_shape'], 'linestyle': formatter['line_style.opaque_shape'], 'facecolor': fc, 'edgecolor': ec}
    if duration is None or isinstance(duration, circuit.ParameterExpression):
        duration = formatter['box_width.opaque_shape']
    box_obj = drawings.BoxData(data_type=types.WaveformType.OPAQUE, channels=channel, xvals=[init_time, init_time + duration], yvals=[-0.5 * formatter['box_height.opaque_shape'], 0.5 * formatter['box_height.opaque_shape']], meta=meta, ignore_scaling=True, styles=box_style)
    fill_objs.append(box_obj)
    func_repr = '{func}({params})'.format(func=pulse_shape, params=', '.join(pnames))
    text_style = {'zorder': formatter['layer.annotate'], 'color': formatter['color.annotate'], 'size': formatter['text_size.annotate'], 'va': 'bottom', 'ha': 'center'}
    text_obj = drawings.TextData(data_type=types.LabelType.OPAQUE_BOXTEXT, channels=channel, xvals=[init_time + 0.5 * duration], yvals=[0.5 * formatter['box_height.opaque_shape']], text=func_repr, ignore_scaling=True, styles=text_style)
    fill_objs.append(text_obj)
    return fill_objs