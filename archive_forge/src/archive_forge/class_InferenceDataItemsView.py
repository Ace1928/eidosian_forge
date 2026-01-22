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
class InferenceDataItemsView(ItemsView[str, xr.Dataset]):
    """ItemsView implementation for InferenceData, to allow it to implement Mapping."""

    def __init__(self, parent: 'InferenceData') -> None:
        """Create a new InferenceDataItemsView from an InferenceData object."""
        self.parent = parent

    def __len__(self) -> int:
        """Return the number of groups in the parent InferenceData."""
        return len(self.parent._groups_all)

    def __iter__(self) -> Iterator[Tuple[str, xr.Dataset]]:
        """Iterate through the groups and corresponding Xarray datasets in the InferenceData."""
        parent = self.parent
        for group in parent._groups_all:
            yield (group, getattr(parent, group))

    def __contains__(self, key: object) -> bool:
        """Return True if the (group, dataset) tuple is present, and False otherwise."""
        parent = self.parent
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        group, dataset = key
        if group not in parent._groups_all:
            return False
        if not isinstance(dataset, xr.Dataset):
            return False
        existing_dataset = getattr(parent, group)
        return existing_dataset.equals(dataset)