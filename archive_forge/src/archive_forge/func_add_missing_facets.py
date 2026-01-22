from __future__ import annotations
import itertools
import types
import typing
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from .._utils import cross_join, match
from ..exceptions import PlotnineError
from ..scales.scales import Scales
from .strips import Strips
def add_missing_facets(data: pd.DataFrame, layout: pd.DataFrame, vars: Sequence[str], facet_vals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add missing facets
    """
    missing_facets = list(set(vars) - set(facet_vals.columns.tolist()))
    if missing_facets:
        to_add = layout.loc[:, missing_facets].drop_duplicates()
        to_add.reset_index(drop=True, inplace=True)
        data_rep = np.tile(np.arange(len(data)), len(to_add))
        facet_rep = np.repeat(np.arange(len(to_add)), len(data))
        data = data.iloc[data_rep, :].reset_index(drop=True)
        facet_vals = facet_vals.iloc[data_rep, :].reset_index(drop=True)
        to_add = to_add.iloc[facet_rep, :].reset_index(drop=True)
        facet_vals = pd.concat([facet_vals, to_add], axis=1, ignore_index=False)
    return (data, facet_vals)