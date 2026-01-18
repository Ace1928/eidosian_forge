import math
from . import _catboost
from .core import CatBoost, CatBoostError
from .utils import _import_matplotlib
def to_polynom_string(model):
    _check_model(model)
    return _catboost.to_polynom_string(model._object)