import copy
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _LGBM_BoosterBestScoreType,
from .callback import _EvalResultDict, record_evaluation
from .compat import (SKLEARN_INSTALLED, LGBMNotFittedError, _LGBMAssertAllFinite, _LGBMCheckArray,
from .engine import train
class LGBMClassifier(_LGBMClassifierBase, LGBMModel):
    """LightGBM classifier."""

    def fit(self, X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType, sample_weight: Optional[_LGBM_WeightType]=None, init_score: Optional[_LGBM_InitScoreType]=None, eval_set: Optional[List[_LGBM_ScikitValidSet]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_LGBM_WeightType]]=None, eval_class_weight: Optional[List[float]]=None, eval_init_score: Optional[List[_LGBM_InitScoreType]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, feature_name: _LGBM_FeatureNameConfiguration='auto', categorical_feature: _LGBM_CategoricalFeatureConfiguration='auto', callbacks: Optional[List[Callable]]=None, init_model: Optional[Union[str, Path, Booster, LGBMModel]]=None) -> 'LGBMClassifier':
        """Docstring is inherited from the LGBMModel."""
        _LGBMAssertAllFinite(y)
        _LGBMCheckClassificationTargets(y)
        self._le = _LGBMLabelEncoder().fit(y)
        _y = self._le.transform(y)
        self._class_map = dict(zip(self._le.classes_, self._le.transform(self._le.classes_)))
        if isinstance(self.class_weight, dict):
            self._class_weight = {self._class_map[k]: v for k, v in self.class_weight.items()}
        self._classes = self._le.classes_
        self._n_classes = len(self._classes)
        if self.objective is None:
            self._objective = None
        if not callable(eval_metric):
            if isinstance(eval_metric, list):
                eval_metric_list = eval_metric
            elif isinstance(eval_metric, str):
                eval_metric_list = [eval_metric]
            else:
                eval_metric_list = []
            if self._n_classes > 2:
                for index, metric in enumerate(eval_metric_list):
                    if metric in {'logloss', 'binary_logloss'}:
                        eval_metric_list[index] = 'multi_logloss'
                    elif metric in {'error', 'binary_error'}:
                        eval_metric_list[index] = 'multi_error'
            else:
                for index, metric in enumerate(eval_metric_list):
                    if metric in {'logloss', 'multi_logloss'}:
                        eval_metric_list[index] = 'binary_logloss'
                    elif metric in {'error', 'multi_error'}:
                        eval_metric_list[index] = 'binary_error'
            eval_metric = eval_metric_list
        valid_sets: Optional[List[_LGBM_ScikitValidSet]] = None
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            valid_sets = []
            for valid_x, valid_y in eval_set:
                if valid_x is X and valid_y is y:
                    valid_sets.append((valid_x, _y))
                else:
                    valid_sets.append((valid_x, self._le.transform(valid_y)))
        super().fit(X, _y, sample_weight=sample_weight, init_score=init_score, eval_set=valid_sets, eval_names=eval_names, eval_sample_weight=eval_sample_weight, eval_class_weight=eval_class_weight, eval_init_score=eval_init_score, eval_metric=eval_metric, feature_name=feature_name, categorical_feature=categorical_feature, callbacks=callbacks, init_model=init_model)
        return self
    _base_doc = LGBMModel.fit.__doc__.replace('self : LGBMModel', 'self : LGBMClassifier')
    _base_doc = _base_doc[:_base_doc.find('group :')] + _base_doc[_base_doc.find('eval_set :'):]
    fit.__doc__ = _base_doc[:_base_doc.find('eval_group :')] + _base_doc[_base_doc.find('eval_metric :'):]

    def predict(self, X: _LGBM_ScikitMatrixLike, raw_score: bool=False, start_iteration: int=0, num_iteration: Optional[int]=None, pred_leaf: bool=False, pred_contrib: bool=False, validate_features: bool=False, **kwargs: Any):
        """Docstring is inherited from the LGBMModel."""
        result = self.predict_proba(X=X, raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration, pred_leaf=pred_leaf, pred_contrib=pred_contrib, validate_features=validate_features, **kwargs)
        if callable(self._objective) or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            class_index = np.argmax(result, axis=1)
            return self._le.inverse_transform(class_index)
    predict.__doc__ = LGBMModel.predict.__doc__

    def predict_proba(self, X: _LGBM_ScikitMatrixLike, raw_score: bool=False, start_iteration: int=0, num_iteration: Optional[int]=None, pred_leaf: bool=False, pred_contrib: bool=False, validate_features: bool=False, **kwargs: Any):
        """Docstring is set after definition, using a template."""
        result = super().predict(X=X, raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration, pred_leaf=pred_leaf, pred_contrib=pred_contrib, validate_features=validate_features, **kwargs)
        if callable(self._objective) and (not (raw_score or pred_leaf or pred_contrib)):
            _log_warning('Cannot compute class probabilities or labels due to the usage of customized objective function.\nReturning raw scores instead.')
            return result
        elif self._n_classes > 2 or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            return np.vstack((1.0 - result, result)).transpose()
    predict_proba.__doc__ = _lgbmmodel_doc_predict.format(description='Return the predicted probability for each class for each sample.', X_shape="numpy array, pandas DataFrame, H2O DataTable's Frame , scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]", output_name='predicted_probability', predicted_result_shape='array-like of shape = [n_samples] or shape = [n_samples, n_classes]', X_leaves_shape='array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]', X_SHAP_values_shape='array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects')

    @property
    def classes_(self) -> np.ndarray:
        """:obj:`array` of shape = [n_classes]: The class label array."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No classes found. Need to call fit beforehand.')
        return self._classes

    @property
    def n_classes_(self) -> int:
        """:obj:`int`: The number of classes."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No classes found. Need to call fit beforehand.')
        return self._n_classes