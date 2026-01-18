import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_stats_information_criterion(models):
    rcParams['stats.information_criterion'] = 'waic'
    df_comp = compare({'model1': models.model_1, 'model2': models.model_2})
    assert 'elpd_waic' in df_comp.columns
    rcParams['stats.information_criterion'] = 'loo'
    df_comp = compare({'model1': models.model_1, 'model2': models.model_2})
    assert 'elpd_loo' in df_comp.columns