from typing import List, Optional, Tuple, Dict, Iterable, overload, Union
from altair import (
from altair.utils._vegafusion_data import get_inline_tables, import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.schemapi import Undefined
Get the Vega datasets that correspond to the provided Altair view names

    Parameters
    ----------
    group : dict
        Top-level Vega spec or nested group mark
    vl_chart_names : list of str
        List of the Vega-Lite
    facet_mapping : dict from (str, tuple of int) to (str, tuple of int)
        The facet mapping produced by get_facet_mapping
    scope : tuple of int
        Scope of the group dictionary within a top-level Vega spec

    Returns
    -------
    dict from str to (str, tuple of int)
        Dict from Altair view names to scoped datasets
    