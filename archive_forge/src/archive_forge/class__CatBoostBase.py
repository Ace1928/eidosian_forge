from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
class _CatBoostBase(object):

    def __init__(self, params):
        init_params = params.copy() if params is not None else {}
        stringify_builtin_metrics(init_params)
        self._init_params = init_params
        if 'thread_count' in self._init_params and self._init_params['thread_count'] == -1:
            self._init_params.pop('thread_count')
        if 'fixed_binary_splits' in self._init_params and self._init_params['fixed_binary_splits'] is None:
            self._init_params['fixed_binary_splits'] = []
        self._object = _CatBoost()

    def __getstate__(self):
        params = self._init_params.copy()
        test_evals = self._object._get_test_evals()
        if test_evals:
            params['_test_evals'] = test_evals
        if self.is_fitted():
            params['__model'] = self._serialize_model()
        for attr in ['_prediction_values_change', '_loss_value_change']:
            if getattr(self, attr, None) is not None:
                params[attr] = getattr(self, attr, None)
        return params

    def __setstate__(self, state):
        if '_object' not in dict(self.__dict__.items()):
            self._object = _CatBoost()
        if '_init_params' not in dict(self.__dict__.items()):
            self._init_params = {}
        if '__model' in state:
            self._load_from_string(state['__model'])
            del state['__model']
        if '_test_eval' in state:
            self._set_test_evals([state['_test_eval']])
            del state['_test_eval']
        if '_test_evals' in state:
            self._set_test_evals(state['_test_evals'])
            del state['_test_evals']
        for attr in ['_prediction_values_change', '_loss_value_change']:
            if attr in state:
                setattr(self, attr, state[attr])
                del state[attr]
        self._init_params.update(state)

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, _):
        state = self.__getstate__()
        model = self.__class__()
        model.__setstate__(state)
        return model

    def __eq__(self, other):
        return self._is_comparable_to(other) and self._object == other._object

    def __ne__(self, other):
        return not self._is_comparable_to(other) or self._object != other._object

    def copy(self):
        return self.__copy__()

    def is_fitted(self):
        return getattr(self, '_random_seed', None) is not None

    def _is_comparable_to(self, rhs):
        if not isinstance(rhs, _CatBoostBase):
            return False
        for side, estimator in [('left', self), ('right', rhs)]:
            if not estimator.is_fitted():
                message = 'The {} argument is not fitted, only fitted models could be compared.'
                raise CatBoostError(message.format(side))
        return True

    def _set_trained_model_attributes(self):
        setattr(self, '_is_fitted_', True)
        setattr(self, '_random_seed', self._object._get_random_seed())
        setattr(self, '_learning_rate', self._object._get_learning_rate())
        setattr(self, '_tree_count', self._object._get_tree_count())
        setattr(self, '_n_features_in', self._object._get_n_features_in())

    def _train(self, train_pool, test_pool, params, allow_clear_pool, init_model):
        self._object._train(train_pool, test_pool, params, allow_clear_pool, init_model._object if init_model else None)
        self._set_trained_model_attributes()

    def _set_test_evals(self, test_evals):
        self._object._set_test_evals(test_evals)

    def get_test_eval(self):
        test_evals = self._object._get_test_evals()
        if len(test_evals) == 0:
            if self.is_fitted():
                raise CatBoostError('The model has been trained without an eval set.')
            else:
                raise CatBoostError('You should train the model first.')
        if len(test_evals) > 1:
            raise CatBoostError("With multiple eval sets use 'get_test_evals()'")
        test_eval = test_evals[0]
        return test_eval[0] if len(test_eval) == 1 else test_eval

    def get_test_evals(self):
        test_evals = self._object._get_test_evals()
        if len(test_evals) == 0:
            if self.is_fitted():
                raise CatBoostError('The model has been trained without an eval set.')
            else:
                raise CatBoostError('You should train the model first.')
        return test_evals

    def get_evals_result(self):
        return self._object._get_metrics_evals()

    def get_best_score(self):
        return self._object._get_best_score()

    def get_best_iteration(self):
        return self._object._get_best_iteration()

    def get_n_features_in(self):
        return self._object._get_n_features_in()

    def _get_float_feature_indices(self):
        return self._object._get_float_feature_indices()

    def _get_cat_feature_indices(self):
        return self._object._get_cat_feature_indices()

    def _get_text_feature_indices(self):
        return self._object._get_text_feature_indices()

    def _get_embedding_feature_indices(self):
        return self._object._get_embedding_feature_indices()

    def _base_predict(self, pool, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type):
        return self._object._base_predict(pool, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)

    def _base_virtual_ensembles_predict(self, pool, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose):
        return self._object._base_virtual_ensembles_predict(pool, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose)

    def _staged_predict_iterator(self, pool, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose):
        return self._object._staged_predict_iterator(pool, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)

    def _leaf_indexes_iterator(self, pool, ntree_start, ntree_end):
        return self._object._leaf_indexes_iterator(pool, ntree_start, ntree_end)

    def _base_calc_leaf_indexes(self, pool, ntree_start, ntree_end, thread_count, verbose):
        return self._object._base_calc_leaf_indexes(pool, ntree_start, ntree_end, thread_count, verbose)

    def _base_eval_metrics(self, pool, metrics_description, ntree_start, ntree_end, eval_period, thread_count, result_dir, tmp_dir):
        metrics_description_list = metrics_description if isinstance(metrics_description, list) else [metrics_description]
        return self._object._base_eval_metrics(pool, metrics_description_list, ntree_start, ntree_end, eval_period, thread_count, result_dir, tmp_dir)

    def _calc_fstr(self, type, pool, reference_data, thread_count, verbose, model_output, shap_mode, interaction_indices, shap_calc_type, sage_n_samples, sage_batch_size, sage_detect_convergence):
        return self._object._calc_fstr(type.name, pool, reference_data, thread_count, verbose, model_output, shap_mode, interaction_indices, shap_calc_type, sage_n_samples, sage_batch_size, sage_detect_convergence)

    def _calc_ostr(self, train_pool, test_pool, top_size, ostr_type, update_method, importance_values_sign, thread_count, verbose):
        return self._object._calc_ostr(train_pool, test_pool, top_size, ostr_type, update_method, importance_values_sign, thread_count, verbose)

    def _base_shrink(self, ntree_start, ntree_end):
        self._object._base_shrink(ntree_start, ntree_end)
        self._set_trained_model_attributes()

    def _base_drop_unused_features(self):
        self._object._base_drop_unused_features()

    def _save_model(self, output_file, format, export_parameters, pool):
        import json
        if self.is_fitted():
            params_string = ''
            if export_parameters:
                params_string = json.dumps(export_parameters, cls=_NumpyAwareEncoder)
            self._object._save_model(output_file, format, params_string, pool)

    def _load_model(self, model_file, format):
        if not isinstance(model_file, PATH_TYPES):
            raise CatBoostError('Invalid fname type={}: must be str() or pathlib.Path().'.format(type(model_file)))
        self._init_params = {}
        self._object._load_model(model_file, format)
        self._set_trained_model_attributes()
        for key, value in iteritems(self._get_params()):
            self._init_params[key] = value

    def _serialize_model(self):
        return self._object._serialize_model()

    def _deserialize_model(self, dump_model_str):
        assert isinstance(dump_model_str, bytes), 'Not bytes passed as argument'
        self._object._deserialize_model(dump_model_str)

    def _load_from_string(self, dump_model_str):
        self._deserialize_model(dump_model_str)
        self._set_trained_model_attributes()

    def _load_from_stream(self, stream):
        self._object._load_from_stream(stream)
        self._set_trained_model_attributes()

    def _sum_models(self, models_base, weights=None, ctr_merge_policy='IntersectingCountersAverage'):
        if weights is None:
            weights = [1.0 for _ in models_base]
        models_inner = [model._object for model in models_base]
        self._object._sum_models(models_inner, weights, ctr_merge_policy)
        setattr(self, '_random_seed', 0)
        setattr(self, '_learning_rate', 0)
        setattr(self, '_tree_count', self._object._get_tree_count())

    def _save_borders(self, output_file):
        if not self.is_fitted():
            raise CatBoostError('There is no trained model to use save_borders(). Use fit() to train model. Then use save_borders().')
        self._object._save_borders(output_file)

    def _get_borders(self):
        if not self.is_fitted():
            raise CatBoostError('There is no trained model to use get_feature_borders(). Use fit() to train model. Then use get_borders().')
        return self._object._get_borders()

    def _get_nan_treatments(self):
        assert self.is_fitted()
        return self._object._get_nan_treatments()

    def _get_params(self):
        params = self._object._get_params()
        init_params = self._init_params.copy()
        for key, value in iteritems(init_params):
            if key not in params:
                params[key] = value
        return params

    @staticmethod
    def _is_classification_objective(loss_function):
        return isinstance(loss_function, str) and is_classification_objective(loss_function)

    @staticmethod
    def _is_regression_objective(loss_function):
        return isinstance(loss_function, str) and is_regression_objective(loss_function)

    @staticmethod
    def _is_multiregression_objective(loss_function):
        return isinstance(loss_function, str) and is_multiregression_objective(loss_function)

    @staticmethod
    def _is_multitarget_objective(loss_function):
        return isinstance(loss_function, str) and is_multitarget_objective(loss_function)

    @staticmethod
    def _is_survivalregression_objective(loss_function):
        return isinstance(loss_function, str) and is_survivalregression_objective(loss_function)

    @staticmethod
    def _is_ranking_objective(loss_function):
        return isinstance(loss_function, str) and is_ranking_metric(loss_function)

    def get_metadata(self):
        return self._object._get_metadata_wrapper()

    @property
    def tree_count_(self):
        return getattr(self, '_tree_count') if self.is_fitted() else None

    @property
    def random_seed_(self):
        return getattr(self, '_random_seed') if self.is_fitted() else None

    @property
    def learning_rate_(self):
        return getattr(self, '_learning_rate') if self.is_fitted() else None

    @property
    def n_features_in_(self):
        return getattr(self, '_n_features_in') if self.is_fitted() else None

    @property
    def feature_names_(self):
        return self._object._get_feature_names() if self.is_fitted() else None

    @property
    def classes_(self):
        return self._object._get_class_labels() if self.is_fitted() else None

    @property
    def evals_result_(self):
        return self.get_evals_result()

    @property
    def best_score_(self):
        return self.get_best_score()

    @property
    def best_iteration_(self):
        return self.get_best_iteration()

    def _get_tree_splits(self, tree_idx, pool):
        return self._object._get_tree_splits(tree_idx, pool)

    def _get_tree_leaf_values(self, tree_idx):
        return self._object._get_tree_leaf_values(tree_idx)

    def _get_tree_step_nodes(self, tree_idx):
        return self._object._get_tree_step_nodes(tree_idx)

    def _get_tree_node_to_leaf(self, tree_idx):
        return self._object._get_tree_node_to_leaf(tree_idx)

    def get_tree_leaf_counts(self):
        """
        Returns
        -------
        tree_leaf_counts : 1d-array of numpy.uint32 of size tree_count_.
        tree_leaf_counts[i] equals to the number of leafs in i-th tree of the ensemble.
        """
        return self._object._get_tree_leaf_counts()

    def get_leaf_values(self):
        """
        Returns
        -------
        leaf_values : 1d-array of leaf values for all trees.
        Value corresponding to j-th leaf of i-th tree is at position
        sum(get_tree_leaf_counts()[:i]) + j (leaf and tree indexing starts from zero).
        """
        return self._object._get_leaf_values()

    def get_leaf_weights(self):
        """
        Returns
        -------
        leaf_weights : 1d-array of leaf weights for all trees.
        Weight of j-th leaf of i-th tree is at position
        sum(get_tree_leaf_counts()[:i]) + j (leaf and tree indexing starts from zero).
        """
        return self._object._get_leaf_weights()

    def set_leaf_values(self, new_leaf_values):
        """
        Sets values at tree leafs of ensemble equal to new_leaf_values.

        Parameters
        ----------
        new_leaf_values : 1d-array with new leaf values for all trees.
        It's size should be equal to sum(get_tree_leaf_counts()).
        Value corresponding to j-th leaf of i-th tree should be at position
        sum(get_tree_leaf_counts()[:i]) + j (leaf and tree indexing starts from zero).
        """
        self._object._set_leaf_values(new_leaf_values)

    def set_feature_names(self, feature_names):
        """
        Sets feature names equal to feature_names

        Parameters
        ----------
        feature_names: 1-d array of strings with new feature names in the same order as in pool
        """
        self._object._set_feature_names(feature_names)

    def _get_tags(self):
        tags = {'requires_positive_X': False, 'requires_positive_y': False, 'requires_y': True, 'poor_score': False, 'no_validation': True, 'stateless': False, 'pairwise': False, 'multilabel': False, '_skip_test': False, 'multioutput_only': False, 'binary_only': False, 'requires_fit': True}
        params = deepcopy(self._init_params)
        if params is None:
            params = {}
        _process_synonyms(params)
        tags['non_deterministic'] = 'task_type' in params and params['task_type'] == 'GPU'
        loss_function = params.get('loss_function', '')
        tags['multioutput'] = loss_function == 'MultiRMSE' or loss_function == 'RMSEWithUncertainty'
        tags['allow_nan'] = 'nan_mode' not in params or params['nan_mode'] != 'Forbidden'
        return tags

    def get_scale_and_bias(self):
        return self._object._get_scale_and_bias()

    def set_scale_and_bias(self, scale, bias):
        if isinstance(bias, FLOAT_TYPES):
            self._object._set_scale_and_bias(scale, [bias])
        else:
            self._object._set_scale_and_bias(scale, bias)