import os
import re
import sys
import uuid
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, Sequence
from copy import copy as ccopy
from copy import deepcopy
import datetime
from html import escape
from typing import (
import numpy as np
import xarray as xr
from packaging import version
from ..rcparams import rcParams
from ..utils import HtmlTemplate, _subset_list, _var_names, either_dict_or_kwargs
from .base import _extend_xr_method, _make_json_serializable, dict_to_dataset
def _group_names(self, groups: Optional[Union[str, List[str]]], filter_groups: Optional["Literal['like', 'regex']"]=None) -> List[str]:
    """Handle expansion of group names input across arviz.

        Parameters
        ----------
        groups: str, list of str or None
            group or metagroup names.
        idata: xarray.Dataset
            Posterior data in an xarray
        filter_groups: {None, "like", "regex"}, optional, default=None
            If `None` (default), interpret groups as the real group or metagroup names.
            If "like", interpret groups as substrings of the real group or metagroup names.
            If "regex", interpret groups as regular expressions on the real group or
            metagroup names. A la `pandas.filter`.

        Returns
        -------
        groups: list
        """
    if filter_groups not in {None, 'like', 'regex'}:
        raise ValueError(f"'filter_groups' can only be None, 'like', or 'regex', got: '{filter_groups}'")
    all_groups = self._groups_all
    if groups is None:
        return all_groups
    if isinstance(groups, str):
        groups = [groups]
    sel_groups = []
    metagroups = rcParams['data.metagroups']
    for group in groups:
        if group[0] == '~':
            sel_groups.extend([f'~{item}' for item in metagroups[group[1:]] if item in all_groups] if group[1:] in metagroups else [group])
        else:
            sel_groups.extend([item for item in metagroups[group] if item in all_groups] if group in metagroups else [group])
    try:
        group_names = _subset_list(sel_groups, all_groups, filter_items=filter_groups)
    except KeyError as err:
        msg = ' '.join(('groups:', f'{err}', 'in InferenceData'))
        raise KeyError(msg) from err
    return group_names