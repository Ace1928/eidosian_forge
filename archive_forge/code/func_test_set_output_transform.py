import os
import pkgutil
import re
import sys
import warnings
from functools import partial
from inspect import isgenerator, signature
from itertools import chain, product
from pathlib import Path
import numpy as np
import pytest
import sklearn
from sklearn.cluster import (
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.experimental import (
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.model_selection import (
from sklearn.neighbors import (
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.utils import _IS_WASM, IS_PYPY, all_estimators
from sklearn.utils._tags import _DEFAULT_TAGS, _safe_tags
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
@pytest.mark.parametrize('estimator', SET_OUTPUT_ESTIMATORS, ids=_get_check_estimator_ids)
def test_set_output_transform(estimator):
    name = estimator.__class__.__name__
    if not hasattr(estimator, 'set_output'):
        pytest.skip(f'Skipping check_set_output_transform for {name}: Does not support set_output API')
    _set_checking_parameters(estimator)
    with ignore_warnings(category=FutureWarning):
        check_set_output_transform(estimator.__class__.__name__, estimator)