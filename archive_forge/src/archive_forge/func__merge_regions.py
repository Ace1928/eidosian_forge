import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
@staticmethod
def _merge_regions(region1, region2, operation):
    if region1 is None or operation == 'overwrite':
        return region2
    data = [d.data for d in region1] + [d.data for d in region2]
    prev = len(data)
    new = None
    while prev != new:
        prev = len(data)
        contiguous = []
        for l, u in data:
            if not util.isfinite(l) or not util.isfinite(u):
                continue
            overlap = False
            for i, (pl, pu) in enumerate(contiguous):
                if l >= pl and l <= pu:
                    pu = max(u, pu)
                    overlap = True
                elif u <= pu and u >= pl:
                    pl = min(l, pl)
                    overlap = True
                if overlap:
                    contiguous[i] = (pl, pu)
            if not overlap:
                contiguous.append((l, u))
        new = len(contiguous)
        data = contiguous
    return NdOverlay([(i, region1.last.clone(l, u)) for i, (l, u) in enumerate(data)])