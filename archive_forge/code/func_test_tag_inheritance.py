import pickle
import re
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
def test_tag_inheritance():
    nan_tag_est = NaNTag()
    no_nan_tag_est = NoNaNTag()
    assert nan_tag_est._get_tags()['allow_nan']
    assert not no_nan_tag_est._get_tags()['allow_nan']
    redefine_tags_est = OverrideTag()
    assert not redefine_tags_est._get_tags()['allow_nan']
    diamond_tag_est = DiamondOverwriteTag()
    assert diamond_tag_est._get_tags()['allow_nan']
    inherit_diamond_tag_est = InheritDiamondOverwriteTag()
    assert inherit_diamond_tag_est._get_tags()['allow_nan']