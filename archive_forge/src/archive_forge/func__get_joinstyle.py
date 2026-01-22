from __future__ import annotations
import typing
from collections import Counter
from contextlib import suppress
from warnings import warn
import numpy as np
from .._utils import SIZE_FACTOR, make_line_segments, match, to_rgba
from ..doctools import document
from ..exceptions import PlotnineWarning
from .geom import geom
def _get_joinstyle(data: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    with suppress(KeyError):
        if params['linejoin'] == 'mitre':
            params['linejoin'] = 'miter'
    with suppress(KeyError):
        if params['lineend'] == 'square':
            params['lineend'] = 'projecting'
    joinstyle = params.get('linejoin', 'miter')
    capstyle = params.get('lineend', 'butt')
    d = {}
    if data['linetype'].iloc[0] == 'solid':
        d['solid_joinstyle'] = joinstyle
        d['solid_capstyle'] = capstyle
    elif data['linetype'].iloc[0] == 'dashed':
        d['dash_joinstyle'] = joinstyle
        d['dash_capstyle'] = capstyle
    return d