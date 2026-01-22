from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
from typing import TYPE_CHECKING
def _legend_artist(self, variables: list[str], value: Any, scales: dict[str, Scale]) -> Artist:
    key = {v: value for v in variables}
    res = self._resolve_properties(key, scales)
    return mpl.collections.PathCollection(paths=[res['path']], sizes=[res['size']], facecolors=[res['facecolor']], edgecolors=[res['edgecolor']], linewidths=[res['linewidth']], linestyles=[res['edgestyle']], transform=mpl.transforms.IdentityTransform(), **self.artist_kws)