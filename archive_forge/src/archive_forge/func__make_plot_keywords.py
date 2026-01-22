from __future__ import annotations
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
    kwds['bw_method'] = self.bw_method
    kwds['ind'] = type(self)._get_ind(y, ind=self.ind)