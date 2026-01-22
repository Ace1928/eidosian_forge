import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from scipy.special import softmax
from ._typing import ArrayLike, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import SKLEARN_INSTALLED, XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .config import config_context
from .core import (
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array, _is_pandas_df
from .training import train
def _wrap_evaluation_matrices(missing: float, X: Any, y: Any, group: Optional[Any], qid: Optional[Any], sample_weight: Optional[Any], base_margin: Optional[Any], feature_weights: Optional[Any], eval_set: Optional[Sequence[Tuple[Any, Any]]], sample_weight_eval_set: Optional[Sequence[Any]], base_margin_eval_set: Optional[Sequence[Any]], eval_group: Optional[Sequence[Any]], eval_qid: Optional[Sequence[Any]], create_dmatrix: Callable, enable_categorical: bool, feature_types: Optional[FeatureTypes]) -> Tuple[Any, List[Tuple[Any, str]]]:
    """Convert array_like evaluation matrices into DMatrix.  Perform validation on the
    way."""
    train_dmatrix = create_dmatrix(data=X, label=y, group=group, qid=qid, weight=sample_weight, base_margin=base_margin, feature_weights=feature_weights, missing=missing, enable_categorical=enable_categorical, feature_types=feature_types, ref=None)
    n_validation = 0 if eval_set is None else len(eval_set)

    def validate_or_none(meta: Optional[Sequence], name: str) -> Sequence:
        if meta is None:
            return [None] * n_validation
        if len(meta) != n_validation:
            raise ValueError(f"{name}'s length does not equal `eval_set`'s length, " + f'expecting {n_validation}, got {len(meta)}')
        return meta
    if eval_set is not None:
        sample_weight_eval_set = validate_or_none(sample_weight_eval_set, 'sample_weight_eval_set')
        base_margin_eval_set = validate_or_none(base_margin_eval_set, 'base_margin_eval_set')
        eval_group = validate_or_none(eval_group, 'eval_group')
        eval_qid = validate_or_none(eval_qid, 'eval_qid')
        evals = []
        for i, (valid_X, valid_y) in enumerate(eval_set):
            if all((valid_X is X, valid_y is y, sample_weight_eval_set[i] is sample_weight, base_margin_eval_set[i] is base_margin, eval_group[i] is group, eval_qid[i] is qid)):
                evals.append(train_dmatrix)
            else:
                m = create_dmatrix(data=valid_X, label=valid_y, weight=sample_weight_eval_set[i], group=eval_group[i], qid=eval_qid[i], base_margin=base_margin_eval_set[i], missing=missing, enable_categorical=enable_categorical, feature_types=feature_types, ref=train_dmatrix)
                evals.append(m)
        nevals = len(evals)
        eval_names = [f'validation_{i}' for i in range(nevals)]
        evals = list(zip(evals, eval_names))
    else:
        if any((meta is not None for meta in [sample_weight_eval_set, base_margin_eval_set, eval_group, eval_qid])):
            raise ValueError('`eval_set` is not set but one of the other evaluation meta info is not None.')
        evals = []
    return (train_dmatrix, evals)