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
class CatBoost(_CatBoostBase):
    """
    CatBoost model. Contains training, prediction and evaluation methods.
    """

    def __init__(self, params=None):
        """
        Initialize the CatBoost.

        Parameters
        ----------
        params : dict
            Parameters for CatBoost.
            If  None, all params are set to their defaults.
            If  dict, overriding parameters present in dict.
        """
        super(CatBoost, self).__init__(params)

    def _dataset_train_eval_split(self, train_pool, params, save_eval_pool):
        """
        returns:
            train_pool, eval_pool
                eval_pool will be uninitialized if save_eval_pool is false
        """
        is_classification = getattr(self, '_estimator_type', None) == 'classifier' or _CatBoostBase._is_classification_objective(params.get('loss_function', 'RMSE'))
        return train_pool.train_eval_split(params.get('has_time', False), is_classification, params['eval_fraction'], save_eval_pool)

    def _prepare_train_params(self, X=None, y=None, cat_features=None, text_features=None, embedding_features=None, pairs=None, sample_weight=None, group_id=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=None, plot_file=None, column_description=None, verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None):
        params = deepcopy(self._init_params)
        if params is None:
            params = {}
        _process_synonyms(params)
        if isinstance(X, FeaturesData):
            warnings.warn('FeaturesData is deprecated for using in fit function and soon will not be supported. If you want to use FeaturesData, please pass it to Pool initialization and use Pool in fit')
        cat_features = _process_feature_indices(cat_features, X, params, 'cat_features')
        text_features = _process_feature_indices(text_features, X, params, 'text_features')
        embedding_features = _process_feature_indices(embedding_features, X, params, 'embedding_features')
        train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description)
        if train_pool.is_empty_:
            raise CatBoostError('X is empty.')
        allow_clear_pool = not isinstance(X, Pool)
        params['loss_function'] = _get_loss_function_for_train(params, getattr(self, '_estimator_type', None), train_pool)
        metric_period, verbose, logging_level = _process_verbose(metric_period, verbose, logging_level, verbose_eval, silent)
        if metric_period is not None:
            params['metric_period'] = metric_period
        if logging_level is not None:
            params['logging_level'] = logging_level
        if verbose is not None:
            params['verbose'] = verbose
        if use_best_model is not None:
            params['use_best_model'] = use_best_model
        if early_stopping_rounds is not None:
            params['od_type'] = 'Iter'
            params['od_wait'] = early_stopping_rounds
            if 'od_pval' in params:
                del params['od_pval']
        if save_snapshot is not None:
            params['save_snapshot'] = save_snapshot
        if snapshot_file is not None:
            params['snapshot_file'] = snapshot_file
        if snapshot_interval is not None:
            params['snapshot_interval'] = snapshot_interval
        if callbacks is not None:
            params['callbacks'] = _TrainCallbacksWrapper(callbacks)
        _check_param_types(params)
        params = _params_type_cast(params)
        _check_train_params(params)
        if params.get('eval_fraction', 0.0) != 0.0:
            if eval_set is not None:
                raise CatBoostError('Both eval_fraction and eval_set specified')
            train_pool, eval_set = self._dataset_train_eval_split(train_pool, params, save_eval_pool=True)
        eval_set_list = eval_set if isinstance(eval_set, list) else [eval_set]
        eval_sets = []
        eval_total_row_count = 0
        for eval_set in eval_set_list:
            if isinstance(eval_set, Pool):
                eval_sets.append(eval_set)
                eval_total_row_count += eval_sets[-1].num_row()
                if eval_sets[-1].num_row() == 0:
                    raise CatBoostError("Empty 'eval_set' in Pool")
            elif isinstance(eval_set, PATH_TYPES):
                eval_sets.append(Pool(eval_set, column_description=column_description))
                eval_total_row_count += eval_sets[-1].num_row()
                if eval_sets[-1].num_row() == 0:
                    raise CatBoostError("Empty 'eval_set' in file {}".format(eval_set))
            elif isinstance(eval_set, tuple):
                if len(eval_set) != 2:
                    raise CatBoostError("Invalid shape of 'eval_set': {}, must be (X, y).".format(str(tuple((type(_) for _ in eval_set)))))
                if eval_set[0] is None or eval_set[1] is None:
                    raise CatBoostError("'eval_set' tuple contains at least one None value")
                eval_sets.append(Pool(eval_set[0], eval_set[1], cat_features=train_pool.get_cat_feature_indices(), text_features=train_pool.get_text_feature_indices(), embedding_features=train_pool.get_embedding_feature_indices()))
                eval_total_row_count += eval_sets[-1].num_row()
                if eval_sets[-1].num_row() == 0:
                    raise CatBoostError("Empty 'eval_set' in tuple")
            elif eval_set is None:
                if len(eval_set_list) > 1:
                    raise CatBoostError('Multiple eval set shall not contain None')
            else:
                raise CatBoostError("Invalid type of 'eval_set': {}, while expected Pool or (X, y) or filename, or list thereof.".format(type(eval_set)))
        if self.get_param('use_best_model') and eval_total_row_count == 0:
            raise CatBoostError("To employ param {'use_best_model': True} provide non-empty 'eval_set'.")
        if init_model is not None and isinstance(init_model, PATH_TYPES):
            try:
                init_model = CatBoost().load_model(init_model)
            except Exception as e:
                raise CatBoostError('Error while loading init_model: {}'.format(e))
        return {'train_pool': train_pool, 'eval_sets': eval_sets, 'params': params, 'allow_clear_pool': allow_clear_pool, 'init_model': init_model}

    def _fit(self, X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout=None, log_cerr=None):
        with log_fixup(log_cout, log_cerr):
            if X is None:
                raise CatBoostError('X must not be None')
            if y is None and (not isinstance(X, PATH_TYPES + (Pool,))):
                raise CatBoostError('y may be None only when X is an instance of catboost.Pool or string')
            train_params = self._prepare_train_params(X=X, y=y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, sample_weight=sample_weight, group_id=group_id, group_weight=group_weight, subgroup_id=subgroup_id, pairs_weight=pairs_weight, baseline=baseline, use_best_model=use_best_model, eval_set=eval_set, verbose=verbose, logging_level=logging_level, plot=plot, plot_file=plot_file, column_description=column_description, verbose_eval=verbose_eval, metric_period=metric_period, silent=silent, early_stopping_rounds=early_stopping_rounds, save_snapshot=save_snapshot, snapshot_file=snapshot_file, snapshot_interval=snapshot_interval, init_model=init_model, callbacks=callbacks)
            params = train_params['params']
            train_pool = train_params['train_pool']
            allow_clear_pool = train_params['allow_clear_pool']
            with plot_wrapper(plot, plot_file, 'Training plots', [_get_train_dir(self.get_params())]):
                self._train(train_pool, train_params['eval_sets'], params, allow_clear_pool, train_params['init_model'])
            loss = self._object._get_loss_function_name()
            if loss and is_groupwise_metric(loss):
                pass
            elif len(self.get_embedding_feature_indices()) > 0:
                pass
            elif not self._object._has_leaf_weights_in_model():
                if allow_clear_pool:
                    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description)
                    if params.get('eval_fraction', 0.0) != 0.0:
                        train_pool, _ = self._dataset_train_eval_split(train_pool, params, save_eval_pool=False)
                self.get_feature_importance(data=train_pool, type=EFstrType.PredictionValuesChange)
            else:
                self.get_feature_importance(type=EFstrType.PredictionValuesChange)
        return self

    def fit(self, X, y=None, cat_features=None, text_features=None, embedding_features=None, pairs=None, sample_weight=None, group_id=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False, plot_file=None, column_description=None, verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None, log_cout=None, log_cerr=None):
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
             or string.
            If not catboost.Pool or catboost.FeaturesData it must be 2 dimensional Feature matrix
             or string - file with dataset.

             Must be non-empty (contain > 0 objects)

        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels of the training data.
            If not None, can be a single- or two- dimensional array with either:
              - numerical values - for regression (including multiregression), ranking and binary classification problems
              - class labels (boolean, integer or string) - for classification (including multiclassification) problems
            Use only if X is not catboost.Pool and does not point to a file.

        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool and not catboost.FeaturesData

        text_features: list or numpy.ndarray, optional (default=None)
            If not none, giving the list of Text columns indices.
            Use only if X is not catboost.Pool and not catboost.FeaturesData

        embedding_features: list or numpy.ndarray, optional (default=None)
            If not none, giving the list of Embedding columns indices.
            Use only if X is not catboost.Pool and not catboost.FeaturesData

        pairs : list or numpy.ndarray or pandas.DataFrame
            The pairs description.
            If list or numpy.ndarrays or pandas.DataFrame, giving 2 dimensional.
            The shape should be Nx2, where N is the pairs' count. The first element of the pair is
            the index of the winner object in the training set. The second element of the pair is
            the index of the loser object in the training set.

        sample_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        group_id : list or numpy.ndarray, optional (default=None)
            group id for each instance.
            If not None, giving 1 dimensional array like data.
            Use only if X is not catboost.Pool.

        group_weight : list or numpy.ndarray, optional (default=None)
            Group weight for each instance.
            If not None, giving 1 dimensional array like data.

        subgroup_id : list or numpy.ndarray, optional (default=None)
            subgroup id for each instance.
            If not None, giving 1 dimensional array like data.
            Use only if X is not catboost.Pool.

        pairs_weight : list or numpy.ndarray, optional (default=None)
            Weight for each pair.
            If not None, giving 1 dimensional array like pairs.

        baseline : list or numpy.ndarray, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not catboost.Pool.

        use_best_model : bool, optional (default=None)
            Flag to use best model

        eval_set : catboost.Pool or list of catboost.Pool or tuple (X, y) or list [(X, y)], optional (default=None)
            Validation dataset or datasets for metrics calculation and possibly early stopping.

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        metric_period : int
            Frequency of evaluating metrics.

        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.

        silent : bool
            If silent is True, logging_level is set to Silent.
            If silent is False, logging_level is set to Verbose.

        verbose_eval : bool or int
            Synonym for verbose. Only one of these parameters should be set.

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error graphs to file

        early_stopping_rounds : int
            Activates Iter overfitting detector with od_wait parameter set to early_stopping_rounds.

        save_snapshot : bool, [default=None]
            Enable progress snapshotting for restoring progress after crashes or interruptions

        snapshot_file : string or pathlib.Path, [default=None]
            Learn progress snapshot file path, if None will use default filename

        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)

        init_model : CatBoost class or string or pathlib.Path, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string or pathlib.Path, load initial model from the path specified by this string.

        callbacks : list, optional (default=None)
            List of callback objects that are applied at end of each iteration.

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        model : CatBoost
        """
        return self._fit(X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

    def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
        if not self.is_fitted() or self.tree_count_ is None:
            raise CatBoostError('There is no trained model to use {}(). Use fit() to train model. Then use this method.'.format(parent_method_name))
        is_single_object = _is_data_single_object(data)
        if not isinstance(data, Pool):
            data = Pool(data=[data] if is_single_object else data, label=label, cat_features=self._get_cat_feature_indices() if not isinstance(data, FeaturesData) else None, text_features=self._get_text_feature_indices() if not isinstance(data, FeaturesData) else None, embedding_features=self._get_embedding_feature_indices() if not isinstance(data, FeaturesData) else None, thread_count=thread_count)
        return (data, is_single_object)

    def _validate_prediction_type(self, prediction_type, valid_prediction_types=('Class', 'RawFormulaVal', 'Probability', 'LogProbability', 'Exponent', 'RMSEWithUncertainty')):
        if not isinstance(prediction_type, STRING_TYPES):
            raise CatBoostError('Invalid prediction_type type={}: must be str().'.format(type(prediction_type)))
        if prediction_type not in valid_prediction_types:
            raise CatBoostError('Invalid value of prediction_type={}: must be {}.'.format(prediction_type, ', '.join(valid_prediction_types)))

    def _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type='CPU'):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
        self._validate_prediction_type(prediction_type)
        predictions = self._base_predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)
        return predictions[0] if data_is_single_object else predictions

    def predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type='CPU'):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return class label.
            - 'Probability' : return probability for every class.
            - 'Exponent' : return Exponent of raw formula value.
            - 'RMSEWithUncertainty': return standard deviation for RMSEWithUncertainty loss function
              (logarithm of the standard deviation is returned by default).

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            If data is for a single object, the return value depends on prediction_type value:
                - 'RawFormulaVal' : return raw formula value.
                - 'Class' : return class label.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of class label for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
        """
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)

    def _virtual_ensembles_predict(self, data, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose, parent_method_name):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
        self._validate_prediction_type(prediction_type, ['VirtEnsembles', 'TotalUncertainty'])
        if ntree_end == 0:
            ntree_end = self.tree_count_
        predictions = self._base_virtual_ensembles_predict(data, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose)
        if prediction_type == 'VirtEnsembles':
            shape = predictions.shape
            predictions = predictions.reshape(shape[0], virtual_ensembles_count, int(shape[1] / virtual_ensembles_count))
        return predictions[0] if data_is_single_object else predictions

    def virtual_ensembles_predict(self, data, prediction_type='VirtEnsembles', ntree_end=0, virtual_ensembles_count=10, thread_count=-1, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'VirtEnsembles': return V (virtual_ensembles_count) predictions.
                k-th virtEnsemle consists of trees [0, T/2] + [T/2 + T/(2V) * k, T/2 + T/(2V) * (k + 1)]  * constant.
            - 'TotalUncertainty': see returned predictions format in 'Returns' part

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        virtual_ensembles_count: int, optional (default=10)
            virtual ensembles count for 'TotalUncertainty' and 'VirtEnsembles' prediction types.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            (with V as virtual_ensembles_count and T as trees count,
            k-th virtEnsemle consists of trees [0, T/2] + [T/2 + T/(2V) * k, T/2 + T/(2V) * (k + 1)]  * constant)
            If data is for a single object, return 1-dimensional array of predictions with size depends on prediction type,
            otherwise return 2-dimensional numpy.ndarray with shape (number_of_objects x size depends on prediction type);
            Returned predictions depends on prediction type:
            If loss-function was RMSEWithUncertainty:
                - 'VirtEnsembles': [mean0, var0, mean1, var1, ..., vark-1].
                - 'TotalUncertainty': [mean_predict, KnowledgeUnc, DataUnc].
            otherwise for regression:
                - 'VirtEnsembles':  [mean0, mean1, ...].
                - 'TotalUncertainty': [mean_predicts, KnowledgeUnc].
            otherwise for binary classification:
                - 'VirtEnsembles':  [ApproxRawFormulaVal0, ApproxRawFormulaVal1, ..., ApproxRawFormulaValk-1].
                - 'TotalUncertainty':  [DataUnc, TotalUnc].
        """
        return self._virtual_ensembles_predict(data, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose, 'virtual_ensembles_predict')

    def _staged_predict(self, data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, parent_method_name):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
        self._validate_prediction_type(prediction_type)
        if ntree_end == 0:
            ntree_end = self.tree_count_
        staged_predict_iterator = self._staged_predict_iterator(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)
        for predictions in staged_predict_iterator:
            yield (predictions[0] if data_is_single_object else predictions)

    def staged_predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Class' : return class label.
            - 'Probability' : return probability for every class.
            - 'RMSEWithUncertainty': return standard deviation for RMSEWithUncertainty loss function
              (logarithm of the standard deviation is returned by default).

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object, the return value depends on prediction_type value:
                - 'RawFormulaVal' : return raw formula value.
                - 'Class' : return class label.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of class label for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
        """
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def _iterate_leaf_indexes(self, data, ntree_start, ntree_end):
        if ntree_end == 0:
            ntree_end = self.tree_count_
        data, _ = self._process_predict_input_data(data, 'iterate_leaf_indexes', thread_count=-1)
        leaf_indexes_iterator = self._leaf_indexes_iterator(data, ntree_start, ntree_end)
        for leaf_index in leaf_indexes_iterator:
            yield leaf_index

    def iterate_leaf_indexes(self, data, ntree_start=0, ntree_end=0):
        """
        Returns indexes of leafs to which objects from pool are mapped by model trees.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Index of first tree for which leaf indexes will be calculated (zero-based indexing).

        ntree_end: int, optional (default=0)
            Index of the tree after last tree for which leaf indexes will be calculated (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        Returns
        -------
        leaf_indexes : generator. For each object in pool yields one-dimensional numpy.ndarray of leaf indexes.
        """
        return self._iterate_leaf_indexes(data, ntree_start, ntree_end)

    def _calc_leaf_indexes(self, data, ntree_start, ntree_end, thread_count, verbose):
        if ntree_end == 0:
            ntree_end = self.tree_count_
        data, _ = self._process_predict_input_data(data, 'calc_leaf_indexes', thread_count)
        return self._base_calc_leaf_indexes(data, ntree_start, ntree_end, thread_count, verbose)

    def calc_leaf_indexes(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=False):
        """
        Returns indexes of leafs to which objects from pool are mapped by model trees.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Index of first tree for which leaf indexes will be calculated (zero-based indexing).

        ntree_end: int, optional (default=0)
            Index of the tree after last tree for which leaf indexes will be calculated (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool (default=False)
            Enable debug logging level.

        Returns
        -------
        leaf_indexes : 2-dimensional numpy.ndarray of numpy.uint32 with shape (object count, ntree_end - ntree_start).
            i-th row is an array of leaf indexes for i-th object.
        """
        return self._calc_leaf_indexes(data, ntree_start, ntree_end, thread_count, verbose)

    def get_cat_feature_indices(self):
        if not self.is_fitted():
            raise CatBoostError('Model is not fitted')
        return self._get_cat_feature_indices()

    def get_text_feature_indices(self):
        if not self.is_fitted():
            raise CatBoostError('Model is not fitted')
        return self._get_text_feature_indices()

    def get_embedding_feature_indices(self):
        if not self.is_fitted():
            raise CatBoostError('Model is not fitted')
        return self._get_embedding_feature_indices()

    def _eval_metrics(self, data, metrics, ntree_start, ntree_end, eval_period, thread_count, res_dir, tmp_dir, plot, plot_file, log_cout=None, log_cerr=None):
        if not self.is_fitted():
            raise CatBoostError('There is no trained model to evaluate metrics on. Use fit() to train model. Then call this method.')
        if not isinstance(data, Pool):
            raise CatBoostError('Invalid data type={}, must be catboost.Pool.'.format(type(data)))
        if data.is_empty_:
            raise CatBoostError('Data is empty.')
        if not isinstance(metrics, ARRAY_TYPES) and (not isinstance(metrics, STRING_TYPES)) and (not isinstance(metrics, BuiltinMetric)):
            raise CatBoostError('Invalid metrics type={}, must be list(), str() or one of builtin catboost.metrics.* class instances.'.format(type(metrics)))
        if not all(map(lambda metric: isinstance(metric, string_types) or isinstance(metric, BuiltinMetric), metrics)):
            raise CatBoostError('Invalid metric type: must be string() or one of builtin catboost.metrics.* class instances.')
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        if isinstance(metrics, STRING_TYPES) or isinstance(metrics, BuiltinMetric):
            metrics = [metrics]
        metrics = stringify_builtin_metrics_list(metrics)
        with log_fixup(log_cout, log_cerr), plot_wrapper(plot, plot_file, 'Eval metrics plot', [res_dir]):
            metrics_score, metric_names = self._base_eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, res_dir, tmp_dir)
        return dict(zip(metric_names, metrics_score))

    def eval_metrics(self, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None, plot=False, plot_file=None, log_cout=None, log_cerr=None):
        """
        Calculate metrics.

        Parameters
        ----------
        data : catboost.Pool
            Data to evaluate metrics on.

        metrics : list of strings or catboost.metrics.BuiltinMetric
            List of evaluated metrics.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        tmp_dir : string or pathlib.Path (default=None)
            The name of the temporary directory for intermediate results.
            If None, then the name will be generated.

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error graphs to file

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        prediction : dict: metric -> array of shape [(ntree_end - ntree_start) / eval_period]
        """
        return self._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, _get_train_dir(self.get_params()), tmp_dir, plot, plot_file, log_cout, log_cerr)

    def compare(self, model, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None, plot_file=None, log_cout=None, log_cerr=None):
        """
        Draw train and eval errors in Jupyter notebook for both models

        Parameters
        ----------
        model: CatBoost model
            Another model to draw metrics

        data : catboost.Pool
            Data to evaluate metrics on.

        metrics : list of strings or catboost.metrics.BuiltinMetric
            List of evaluated metrics.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        tmp_dir : string or pathlib.Path (default=None)
            The name of the temporary directory for intermediate results.
            If None, then the name will be generated.

        plot_file : file-like or str, optional (default=None)
            If not None, save eval error graphs to file

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used
        """
        if model is None:
            raise CatBoostError('You should provide model for comparison.')
        if data is None:
            raise CatBoostError('You should provide data for comparison.')
        if metrics is None:
            raise CatBoostError('You should provide metrics for comparison.')
        need_to_remove = False
        if tmp_dir is None:
            need_to_remove = True
            tmp_dir = tempfile.mkdtemp()
        first_dir = os.path.join(tmp_dir, 'first_model')
        second_dir = os.path.join(tmp_dir, 'second_model')
        create_dir_if_not_exist(tmp_dir)
        create_dir_if_not_exist(first_dir)
        create_dir_if_not_exist(second_dir)
        with plot_wrapper(True, plot_file=plot_file, plot_title='Compare models', train_dirs=[first_dir, second_dir]):
            self._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, first_dir, tmp_dir, plot=False, plot_file=None, log_cout=log_cout, log_cerr=log_cerr)
            model._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, second_dir, tmp_dir, plot=False, plot_file=None, log_cout=log_cout, log_cerr=log_cerr)
        if need_to_remove:
            shutil.rmtree(tmp_dir)

    def create_metric_calcer(self, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None):
        """
        Create batch metric calcer. Could be used to aggregate metric on several pools
        Parameters
        ----------
            Same as in eval_metrics except data
        Returns
        -------
            BatchMetricCalcer object

        Usage example
        -------
        # Large dataset is partitioned into parts [part1, part2]
        model.fit(params)
        batch_calcer = model.create_metric_calcer(['Logloss'])
        batch_calcer.add(part1)
        batch_calcer.add(part2)
        metrics = batch_calcer.eval_metrics()
        """
        if not self.is_fitted():
            raise CatBoostError('There is no trained model to evaluate metrics on. Use fit() to train model. Then call this method.')
        return BatchMetricCalcer(self._object, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir)

    @property
    def feature_importances_(self):
        loss = self._object._get_loss_function_name()
        if loss and is_groupwise_metric(loss):
            return np.array(getattr(self, '_loss_value_change', None))
        else:
            return np.array(getattr(self, '_prediction_values_change', None))

    def get_feature_importance(self, data=None, type=EFstrType.FeatureImportance, prettified=False, thread_count=-1, verbose=False, fstr_type=None, shap_mode='Auto', model_output='Raw', interaction_indices=None, shap_calc_type='Regular', reference_data=None, sage_n_samples=128, sage_batch_size=512, sage_detect_convergence=True, log_cout=None, log_cerr=None):
        """
        Parameters
        ----------
        data :
            Data to get feature importance.
            If type in ('LossFunctionChange', 'ShapValues', 'ShapInteractionValues') data must of Pool type.
                For every object in this dataset feature importances will be calculated.
            if type == 'SageValues' data must of Pool type.
                For every feature in this dataset importance will be calculated.
            If type == 'PredictionValuesChange', data is None or a dataset of Pool type
                Dataset specification is needed only in case if the model does not contain leaf weight information (trained with CatBoost v < 0.9).
            If type == 'PredictionDiff' data must contain a matrix of feature values of shape (2, n_features).
                Possible types are catboost.Pool or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData or pandas.SparseDataFrame or scipy.sparse.spmatrix
            If type == 'FeatureImportance'
                See 'PredictionValuesChange' for non-ranking metrics and 'LossFunctionChange' for ranking metrics.
            If type == 'Interaction'
                This parameter is not used.

        type : EFstrType or string (converted to EFstrType), optional
                    (default=EFstrType.FeatureImportance)
            Possible values:
                - PredictionValuesChange
                    Calculate score for every feature.
                - LossFunctionChange
                    Calculate score for every feature by loss.
                - FeatureImportance
                    PredictionValuesChange for non-ranking metrics and LossFunctionChange for ranking metrics
                - ShapValues
                    Calculate SHAP Values for every object.
                - ShapInteractionValues
                    Calculate SHAP Interaction Values between each pair of features for every object
                - Interaction
                    Calculate pairwise score between every feature.
                - PredictionDiff
                    Calculate most important features explaining difference in predictions for a pair of documents.
                - SageValues
                    Calculate SAGE value for every feature

        prettified : bool, optional (default=False)
            change returned data format to the list of (feature_id, importance) pairs sorted by importance

        thread_count : int, optional (default=-1)
            Number of threads.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool or int
            If False, then evaluation is not logged. If True, then each possible iteration is logged.
            If a positive integer, then it stands for the size of batch N. After processing each batch, print progress
            and remaining time.

        fstr_type : string, deprecated, use type instead

        shap_mode : string, optional (default="Auto")
            used only for ShapValues type
            Possible values:
                - "Auto"
                    Use direct SHAP Values calculation only if data size is smaller than average leaves number
                    (the best of two strategies below is chosen).
                - "UsePreCalc"
                    Calculate SHAP Values for every leaf in preprocessing. Final complexity is
                    O(NT(D+F))+O(TL^2 D^2) where N is the number of documents(objects), T - number of trees,
                    D - average tree depth, F - average number of features in tree, L - average number of leaves in tree
                    This is much faster (because of a smaller constant) than direct calculation when N >> L
                - "NoPreCalc"
                    Use direct SHAP Values calculation calculation with complexity O(NTLD^2). Direct algorithm
                    is faster when N < L (algorithm from https://arxiv.org/abs/1802.03888)

        shap_calc_type : EShapCalcType or string, optional (default="Regular")
            used only for ShapValues type
            Possible values:
                - "Regular"
                    Calculate regular SHAP values
                - "Approximate"
                    Calculate approximate SHAP values
                - "Exact"
                    Calculate exact SHAP values

        interaction_indices : list of int or string (feature_idx_1, feature_idx_2), optional (default=None)
            used only for ShapInteractionValues type
            Calculate SHAP Interaction Values between pair of features feature_idx_1 and feature_idx_2 for every object

        reference_data: catboost.Pool or None
            Reference data for Independent Tree SHAP values from https://arxiv.org/abs/1905.04610v1
            if type == 'ShapValues' and reference_data is not None, then Independent Tree SHAP values are calculated

        sage_n_samples: int, optional (default=32)
            Number of outer samples used in SAGE values approximation algorithm
        sage_batch_size: int, optional (default=min(512, number of samples in dataset))
            Number of samples used on each step of SAGE values approximation algorithm
        sage_detect_convergence: bool, optional (default=False)
            If set True, sage values calculation will be stopped either when sage values converge
            or when sage_n_samples iterations of algorithm pass

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        depends on type:
            - FeatureImportance
                See PredictionValuesChange for non-ranking metrics and LossFunctionChange for ranking metrics.
            - PredictionValuesChange, LossFunctionChange, PredictionDiff, SageValues with prettified=False (default)
                list of length [n_features] with feature_importance values (float) for feature
            - PredictionValuesChange, LossFunctionChange, PredictionDiff, SageValues with prettified=True
                list of length [n_features] with (feature_id (string), feature_importance (float)) pairs, sorted by feature_importance in descending order
            - ShapValues
                np.ndarray of shape (n_objects, n_features + 1) with Shap values (float) for (object, feature).
                In case of multiclass the returned value is np.ndarray of shape
                (n_objects, classes_count, n_features + 1). For each object it contains Shap values (float).
                Values are calculated for RawFormulaVal predictions.
            - ShapInteractionValues
                np.ndarray of shape (n_objects, n_features + 1, n_features + 1) with Shap interaction values (float) for (object, feature(i), feature(j)).
                In case of multiclass the returned value is np.ndarray of shape
                (n_objects, classes_count, n_features + 1, n_features + 1). For each object it contains Shap interaction values (float).
                Values are calculated for RawFormulaVal predictions.
            - Interaction
                list of length [n_features] of 3-element lists of (first_feature_index, second_feature_index, interaction_score (float))
        """
        with log_fixup(log_cout, log_cerr):
            if not isinstance(verbose, bool) and (not isinstance(verbose, int)):
                raise CatBoostError('verbose should be bool or int.')
            verbose = int(verbose)
            if verbose < 0:
                raise CatBoostError('verbose should be non-negative.')
            if fstr_type is not None:
                type = fstr_type
            type = enum_from_enum_or_str(EFstrType, type)
            if type == EFstrType.FeatureImportance:
                loss = self._object._get_loss_function_name()
                if loss and is_groupwise_metric(loss):
                    type = EFstrType.LossFunctionChange
                else:
                    type = EFstrType.PredictionValuesChange
            if type == EFstrType.PredictionDiff:
                data, _ = self._process_predict_input_data(data, 'get_feature_importance', thread_count)
                if data.num_row() != 2:
                    raise CatBoostError('{} requires a pair of documents, found {}'.format(type, data.num_row()))
            elif data is not None and (not isinstance(data, Pool)):
                raise CatBoostError('Invalid data type={}, must be catboost.Pool.'.format(_typeof(data)))
            need_meta_info = type == EFstrType.PredictionValuesChange
            empty_data_is_ok = need_meta_info and self._object._has_leaf_weights_in_model() or type == EFstrType.Interaction
            if not empty_data_is_ok:
                if data is None:
                    if need_meta_info:
                        raise CatBoostError('Model has no meta information needed to calculate feature importances.                             Pass training dataset to this function.')
                    else:
                        raise CatBoostError('Feature importance type {} requires training dataset                             to be passed to this function.'.format(type))
                if data.is_empty_:
                    raise CatBoostError('data is empty.')
            shap_calc_type = enum_from_enum_or_str(EShapCalcType, shap_calc_type).value
            fstr, feature_names = self._calc_fstr(type, data, reference_data, thread_count, verbose, model_output, shap_mode, interaction_indices, shap_calc_type, sage_n_samples, sage_batch_size, sage_detect_convergence)
            if type in (EFstrType.PredictionValuesChange, EFstrType.LossFunctionChange, EFstrType.PredictionDiff, EFstrType.SageValues):
                feature_importances = [value[0] for value in fstr]
                attribute_name = None
                if type == EFstrType.PredictionValuesChange:
                    attribute_name = '_prediction_values_change'
                if type == EFstrType.LossFunctionChange:
                    attribute_name = '_loss_value_change'
                if attribute_name:
                    setattr(self, attribute_name, feature_importances)
                if prettified:
                    feature_importances = sorted(zip(feature_names, feature_importances), key=itemgetter(1), reverse=True)
                    columns = ['Feature Id', 'Importances']
                    return DataFrame(feature_importances, columns=columns)
                else:
                    return np.array(feature_importances)
            elif type == EFstrType.ShapValues:
                if isinstance(fstr[0][0], ARRAY_TYPES):
                    return np.array([np.array([np.array([value for value in dimension]) for dimension in doc]) for doc in fstr])
                else:
                    result = [[value for value in doc] for doc in fstr]
                    if prettified:
                        return DataFrame(result)
                    else:
                        return np.array(result)
            elif type == EFstrType.ShapInteractionValues:
                if isinstance(fstr[0][0], ARRAY_TYPES):
                    return np.array([np.array([np.array([feature2 for feature2 in feature1]) for feature1 in doc]) for doc in fstr])
                else:
                    return np.array([np.array([np.array([np.array([feature2 for feature2 in feature1]) for feature1 in dimension]) for dimension in doc]) for doc in fstr])
            elif type == EFstrType.Interaction:
                result = [[int(row[0]), int(row[1]), row[2]] for row in fstr]
                if prettified:
                    columns = ['First Feature Index', 'Second Feature Index', 'Interaction']
                    return DataFrame(result, columns=columns)
                else:
                    return np.array(result)

    def get_object_importance(self, pool, train_pool, top_size=-1, type='Average', update_method='SinglePoint', importance_values_sign='All', thread_count=-1, verbose=False, ostr_type=None, log_cout=None, log_cerr=None):
        """
        This is the implementation of the LeafInfluence algorithm from the following paper:
        https://arxiv.org/pdf/1802.06640.pdf

        Parameters
        ----------
        pool : Pool
            The pool for which you want to evaluate the object importances.

        train_pool : Pool
            The pool on which the model has been trained.

        top_size : int (default=-1)
            Method returns the result of the top_size most important train objects.
            If -1, then the top size is not limited.

        type : string, optional (default='Average')
            Possible values:
                - Average (Method returns the mean train objects scores for all input objects)
                - PerObject (Method returns the train objects scores for every input object)

        importance_values_sign : string, optional (default='All')
            Method returns only Positive, Negative or All values.
            Possible values:
                - Positive
                - Negative
                - All

        update_method : string, optional (default='SinglePoint')
            Possible values:
                - SinglePoint
                - TopKLeaves (It is posible to set top size : TopKLeaves:top=2)
                - AllPoints
            Description of the update set methods are given in section 3.1.3 of the paper.

        thread_count : int, optional (default=-1)
            Number of threads.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool or int
            If False, then evaluation is not logged. If True, then each possible iteration is logged.
            If a positive integer, then it stands for the size of batch N. After processing each batch, print progress
            and remaining time.

        ostr_type : string, deprecated, use type instead

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        object_importances : tuple of two arrays (indices and scores) of shape = [top_size]
        """
        if self.is_fitted() and (not self._object._is_oblivious()):
            raise CatBoostError('Object importance is not supported for non symmetric trees')
        if not isinstance(verbose, bool) and (not isinstance(verbose, int)):
            raise CatBoostError('verbose should be bool or int.')
        verbose = int(verbose)
        if verbose < 0:
            raise CatBoostError('verbose should be non-negative.')
        if ostr_type is not None:
            type = ostr_type
            warnings.warn("'ostr_type' parameter will be deprecated soon, use 'type' parameter instead")
        with log_fixup(log_cout, log_cerr):
            result = self._calc_ostr(train_pool, pool, top_size, type, update_method, importance_values_sign, thread_count, verbose)
        return result

    def shrink(self, ntree_end, ntree_start=0):
        """
        Shrink the model.

        Parameters
        ----------
        ntree_end: int
            Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
        ntree_start: int, optional (default=0)
            Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
        """
        if ntree_start > ntree_end:
            raise CatBoostError('ntree_start should be less than ntree_end.')
        self._base_shrink(ntree_start, ntree_end)

    def drop_unused_features(self):
        """
        Drop unused features information from model
        """
        self._base_drop_unused_features()

    def save_model(self, fname, format='cbm', export_parameters=None, pool=None):
        """
        Save the model to a file.

        Parameters
        ----------
        fname : string
            Output file name.
        format : string
            Possible values:
                * 'cbm' for catboost binary format,
                * 'coreml' to export into Apple CoreML format
                * 'onnx' to export into ONNX-ML format
                * 'pmml' to export into PMML format
                * 'cpp' to export as C++ code
                * 'python' to export as Python code.
        export_parameters : dict
            Parameters for CoreML export:
                * prediction_type : string - either 'probability' or 'raw'
                * coreml_description : string
                * coreml_model_version : string
                * coreml_model_author : string
                * coreml_model_license: string
            Parameters for PMML export:
                * pmml_copyright : string
                * pmml_description : string
                * pmml_model_version : string
        pool : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series or catboost.FeaturesData
            Training pool.
        """
        if not self.is_fitted():
            raise CatBoostError('There is no trained model to use save_model(). Use fit() to train model. Then use this method.')
        if not isinstance(fname, PATH_TYPES):
            raise CatBoostError('Invalid fname type={}: must be str() or pathlib.Path().'.format(type(fname)))
        if pool is not None and (not isinstance(pool, Pool)):
            pool = Pool(data=pool, cat_features=self._get_cat_feature_indices() if not isinstance(pool, FeaturesData) else None, text_features=self._get_text_feature_indices() if not isinstance(pool, FeaturesData) else None, embedding_features=self._get_embedding_feature_indices() if not isinstance(pool, FeaturesData) else None)
        self._save_model(fname, format, export_parameters, pool)

    def load_model(self, fname=None, format='cbm', stream=None, blob=None):
        """
        Load model from a file, stream or blob.

        Parameters
        ----------
        fname : string
            Input file name.
        """
        if (fname is None) + (stream is None) + (blob is None) != 2:
            raise CatBoostError("Exactly one of fname/stream/blob arguments mustn't be None")
        if fname is not None:
            self._load_model(fname, format)
        elif stream is not None:
            self._load_from_stream(stream)
        elif blob is not None:
            self._load_from_string(blob)
        return self

    def get_param(self, key):
        """
        Get param value from CatBoost model.

        Parameters
        ----------
        key : string
            The key to get param value from.

        Returns
        -------
        value :
            The param value of the key, returns None if param do not exist.
        """
        params = self.get_params()
        if params is None:
            return {}
        return params.get(key)

    def get_params(self, deep=True):
        """
        Get all params from CatBoost model.

        Returns
        -------
        result : dict
            Dictionary of {param_key: param_value}.
        """
        params = self._init_params.copy()
        if deep:
            return deepcopy(params)
        else:
            return params

    def get_all_params(self):
        """
        Get all params (specified by user and default params) that were set in training from CatBoost model.
        Full parameters documentation could be found here: https://catboost.ai/docs/concepts/python-reference_parameters-list.html

        Returns
        -------
        result : dict
            Dictionary of {param_key: param_value}.
        """
        if not self.is_fitted():
            raise CatBoostError('There is no trained model to use get_all_params(). Use fit() to train model. Then use this method.')
        return self._object._get_plain_params()

    def save_borders(self, fname):
        """
        Save the model borders to a file.

        Parameters
        ----------
        fname : string or pathlib.Path
            Output file name.
        """
        if not isinstance(fname, PATH_TYPES):
            raise CatBoostError('Invalid fname type={}: must be str() or pathlib.Path().'.format(type(fname)))
        self._save_borders(fname)

    def get_borders(self):
        """
        Return map feature_index: borders for float features.

        """
        return self._get_borders()

    def set_params(self, **params):
        """
        Set parameters into CatBoost model.

        Parameters
        ----------
        **params : key=value format
            List of key=value paris. Example: model.set_params(iterations=500, thread_count=2).
        """
        if self.is_fitted():
            raise CatBoostError("You can't change params of fitted model.")
        for key, value in iteritems(params):
            self._init_params[key] = value
        if 'thread_count' in self._init_params and self._init_params['thread_count'] == -1:
            self._init_params.pop('thread_count')
        return self

    def plot_predictions(self, data, features_to_change, plot=True, plot_file=None):
        """
        To use this function, you should install plotly.

        data: numpy.ndarray or pandas.DataFrame or catboost.Pool
        features_to_change: list-like with int (for indices) or str (for names) elements
            Numerical features indices or names in `data` for which you want to vary prediction value.
        plot: bool
            Plot predictions.
        plot_file: str
            Output file for plot predictions.
        Returns
        -------
            List of list of predictions for all buckets for all samples in data
        """

        def predict(doc, feature_idx, borders, nan_treatment):
            left_extend_border = min(2 * borders[0], -1)
            right_extend_border = max(2 * borders[-1], 1)
            extended_borders = [left_extend_border] + borders + [right_extend_border]
            points = []
            predictions = []
            border_idx = None
            if np.isnan(doc[feature_idx]):
                border_idx = len(borders) if nan_treatment == 'AsTrue' else 0
            for i in range(len(extended_borders) - 1):
                points += [(extended_borders[i] + extended_borders[i + 1]) / 2.0]
                if border_idx is None and doc[feature_idx] < extended_borders[i + 1]:
                    border_idx = i
                buf = doc[feature_idx]
                doc[feature_idx] = points[-1]
                predictions += [self.predict(doc)]
                doc[feature_idx] = buf
            if border_idx is None:
                border_idx = len(borders)
            return (predictions, border_idx)

        def get_layout(go, feature, xaxis):
            return go.Layout(title="Prediction variation for feature '{}'".format(feature), yaxis={'title': 'Prediction', 'side': 'left', 'overlaying': 'y2'}, xaxis=xaxis)
        try:
            import plotly.graph_objs as go
        except ImportError as e:
            warnings.warn('To draw plots you should install plotly.')
            raise ImportError(str(e))
        model_borders = self._get_borders()
        data, _ = self._process_predict_input_data(data, 'vary_feature_value_and_apply', thread_count=-1)
        figs = []
        all_predictions = [{}] * data.num_row()
        nan_treatments = self._get_nan_treatments()
        for feature in features_to_change:
            if not isinstance(feature, int):
                if self.feature_names_ is None or feature not in self.feature_names_:
                    raise CatBoostError('No feature named "{}" in model'.format(feature))
                feature_idx = self.feature_names_.index(feature)
            else:
                feature_idx = feature
                feature = self.feature_names_[feature_idx]
            assert feature_idx in model_borders, 'only float features indexes are supported'
            borders = model_borders[feature_idx]
            if len(borders) == 0:
                xaxis = go.layout.XAxis(title='Bins', tickvals=[0])
                figs += go.Figure(data=[], layout=get_layout(go, feature_idx, xaxis))
            xaxis = go.layout.XAxis(title='Bins', tickmode='array', tickvals=list(range(len(borders) + 1)), ticktext=['(-inf, {:.4f}]'.format(borders[0])] + ['({:.4f}, {:.4f}]'.format(val_1, val_2) for val_1, val_2 in zip(borders[:-1], borders[1:])] + ['({:.4f}, +inf)'.format(borders[-1])], showticklabels=False)
            trace = []
            for idx, features in enumerate(data.get_features()):
                predictions, border_idx = predict(features, feature_idx, borders, nan_treatments[feature_idx])
                all_predictions[idx][feature_idx] = predictions
                trace.append(go.Scatter(y=predictions, mode='lines+markers', name=u'Document {} predictions'.format(idx)))
                trace.append(go.Scatter(x=[border_idx], y=[predictions[border_idx]], showlegend=False))
            layout = get_layout(go, feature, xaxis)
            figs += [go.Figure(data=trace, layout=layout)]
        if plot:
            try_plot_offline(figs)
        if plot_file:
            save_plot_file(plot_file, 'Predictions for all buckets', figs)
        return (all_predictions, figs)

    def plot_partial_dependence(self, data, features, plot=True, plot_file=None, thread_count=-1):
        """
        To use this function, you should install plotly.
        data: numpy.ndarray or pandas.DataFrame or catboost.Pool
        features: int, str, list<int>, tuple<int>, list<string>, tuple<string>
            Float features to calculate partial dependence for. Number of features should be 1 or 2.
        plot: bool
            Plot predictions.
        plot_file: str
            Output file for plot predictions.
        thread_count: int
            Number of threads to use. If -1 use maximum available number of threads.
        Returns
        -------
            If number of features is one - 1d numpy array and figure with line plot.
            If number of features is two - 2d numpy array and figure with 2d heatmap.
        """
        try:
            import plotly.graph_objs as go
        except ImportError as e:
            warnings.warn('To draw plots you should install plotly.')
            raise ImportError(str(e))

        def getFeatureIdx(feature):
            if not isinstance(feature, int):
                if self.feature_names_ is None or feature not in self.feature_names_:
                    raise CatBoostError('No feature named "{}" in model'.format(feature))
                feature_idx = self.feature_names_.index(feature)
            else:
                feature_idx = feature
            assert feature_idx in self._get_borders(), 'only float features indexes are supported'
            assert len(self._get_borders()[feature_idx]) > 0, 'feature with idx {} is not used in model'.format(feature_idx)
            return feature_idx

        def getFeatureIndices(features):
            if isinstance(features, list) or isinstance(features, tuple):
                features_idxs = [getFeatureIdx(feature) for feature in features]
            elif isinstance(features, int) or isinstance(features, str):
                features_idxs = [getFeatureIdx(features)]
            else:
                raise CatBoostError("Unsupported type for argument 'features'. Must be one of: int, string, list<string>, list<int>, tuple<int>, tuple<string>")
            return features_idxs

        def getAxisParams(borders, feature_name=None):
            return {'title': 'Bins' if feature_name is None else "Bins of feature '{}'".format(feature_name), 'tickmode': 'array', 'tickvals': list(range(len(borders) + 1)), 'ticktext': ['(-inf, {:.4f}]'.format(borders[0])] + ['({:.4f}, {:.4f}]'.format(val_1, val_2) for val_1, val_2 in zip(borders[:-1], borders[1:])] + ['({:.4f}, +inf)'.format(borders[-1])], 'showticklabels': False}

        def plot2d(feature_names, borders, predictions):
            xaxis = go.layout.XAxis(**getAxisParams(borders[1], feature_name=feature_names[1]))
            yaxis = go.layout.YAxis(**getAxisParams(borders[0], feature_name=feature_names[0]))
            layout = go.Layout(title='Partial dependence plot for features {}'.format("'{}'".format("', '".join(map(str, feature_names)))), yaxis=yaxis, xaxis=xaxis)
            fig = go.Figure(data=go.Heatmap(z=predictions), layout=layout)
            return fig

        def plot1d(feature, borders, predictions):
            xaxis = go.layout.XAxis(**getAxisParams(borders))
            yaxis = {'title': 'Mean Prediction', 'side': 'left'}
            layout = go.Layout(title="Partial dependence plot for feature '{}'".format(feature), yaxis=yaxis, xaxis=xaxis)
            fig = go.Figure(data=go.Scatter(y=predictions, mode='lines+markers'), layout=layout)
            return fig
        features_idx = getFeatureIndices(features)
        borders = [self._get_borders()[idx] for idx in features_idx]
        if len(features_idx) not in [1, 2]:
            raise CatBoostError("Number of 'features' should be 1 or 2, got {}".format(len(features_idx)))
        is_2d_plot = len(features_idx) == 2
        data, _ = self._process_predict_input_data(data, 'plot_partial_dependence', thread_count=thread_count)
        all_predictions = np.array(self._object._calc_partial_dependence(data, features_idx, thread_count))
        if is_2d_plot:
            all_predictions = all_predictions.reshape([len(x) + 1 for x in borders])
            fig = plot2d(features_idx, borders, all_predictions)
        else:
            fig = plot1d(features_idx[0], borders[0], all_predictions)
        if plot:
            try_plot_offline(fig)
        if plot_file:
            save_plot_file(plot_file, "Partial dependence plot for features '{}'".format(features), fig)
        return (all_predictions, fig)

    def calc_feature_statistics(self, data, target=None, feature=None, prediction_type=None, cat_feature_values=None, plot=True, max_cat_features_on_plot=10, thread_count=-1, plot_file=None):
        """
        Get statistics for the feature using the model, dataset and target.
        To use this function, you should install plotly.

        The catboost model has borders for the float features used in it. The borders divide
        feature values into bins, and the model's prediction depends on the number of the bin where the
        feature value falls in.

        For float features this function takes model's borders and computes
        1) Mean target value for every bin;
        2) Mean model prediction for every bin;
        3) The number of objects in dataset which fall into each bin;
        4) Predictions on varying feature. For every object, varies the feature value
        so that it falls into bin #0, bin #1, ... and counts model predictions.
        Then counts average prediction for each bin.

        For categorical features (only one-hot supported) does the same, but takes feature values
        provided in cat_feature_values instead of borders.

        Parameters
        ----------
        data: numpy.ndarray or pandas.DataFrame or catboost. Pool or dict {'pool_name': pool} if you want several pools
            Data to compute statistics on
        target: numpy.ndarray or pandas.Series or dict {'pool_name': target} if you want several pools or None
            Target corresponding to data
            Use only if data is not catboost.Pool.
        feature: None, int, string, or list of int or strings
            Features indexes or names in pd.DataFrame for which you want to get statistics.
            None, if you need statistics for all features.
        prediction_type: str
            Prediction type used for counting mean_prediction: 'Class', 'Probability' or 'RawFormulaVal'.
            If not specified, is derived from the model.
        cat_feature_values: list or numpy.ndarray or pandas.Series or
                            dict: int or string to list or numpy.ndarray or pandas.Series
            Contains categorical feature values you need to get statistics on.
            Use dict, when parameter 'feature' is a list to specify cat values for different features.
            When parameter 'feature' is int or str, you can just pass list of cat values.
        plot: bool
            Plot statistics.
        max_cat_features_on_plot: int
            If categorical feature takes more than max_cat_features_on_plot different unique values,
            output result on several plots, not more than max_cat_features_on_plot feature values on each.
            Used only if plot=True or plot_file is not None.
        thread_count: int
            Number of threads to use for getting statistics.
        plot_file: str
            Output file for plot statistics.

        Returns
        -------
        dict if parameter 'feature' is int or string, else dict of dicts:
            For each unique feature contain
            python dict with binarized feature statistics.
            For float feature, includes
                    'borders' -- borders for the specified feature in model
                    'binarized_feature' -- numbers of bins where feature values fall
                    'mean_target' -- mean value of target over each bin
                    'mean_prediction' -- mean value of model prediction over each bin
                    'objects_per_bin' -- number of objects per bin
                    'predictions_on_varying_feature' -- averaged over dataset predictions for
                    varying feature (see above)
            For one-hot feature, returns the same, but with 'cat_values' instead of 'borders'
        """
        target_is_none = target is None
        if not isinstance(data, dict):
            data = {'': data}
        if not isinstance(target, dict):
            target = {'': target}
        assert target_is_none or len(data) == len(target), 'inconsistent size of data and target'
        assert target_is_none or target.keys() == data.keys(), 'inconsistent pool_names of data and target'
        for key in data.keys():
            data[key], _ = self._process_predict_input_data(data[key], 'get_binarized_statistics', thread_count, target.get(key, None))
        data, pool_names = (list(data.values()), list(data.keys()))
        if prediction_type is None:
            prediction_type = 'Probability' if self.get_param('loss_function') in ['CrossEntropy', 'Logloss'] else 'RawFormulaVal'
        if prediction_type not in ['Class', 'Probability', 'RawFormulaVal', 'Exponent']:
            raise CatBoostError('Unknown prediction type "{}"'.format(prediction_type))
        if feature is None:
            feature = self.feature_names_
        if cat_feature_values is None:
            cat_feature_values = {}
        elif not isinstance(cat_feature_values, dict):
            if isinstance(feature, list):
                raise CatBoostError('cat_feature_values should be dict when features is a list')
            else:
                cat_feature_values = {feature: cat_feature_values}
        if isinstance(feature, str) or isinstance(feature, int):
            features = [feature]
            is_for_one_feature = True
        else:
            features = feature
            is_for_one_feature = False
        cat_features_nums = []
        float_features_nums = []
        feature_type_mapper = []
        feature_names = []
        feature_name_to_num = {}
        for feature in features:
            if not isinstance(feature, int):
                if self.feature_names_ is None or feature not in self.feature_names_:
                    raise CatBoostError('No feature named "{}" in model'.format(feature))
                feature_num = self.feature_names_.index(feature)
            else:
                feature_num = feature
                feature = self.feature_names_[feature_num]
            if feature in feature_names:
                continue
            if feature_num in cat_feature_values:
                cat_feature_values[feature] = cat_feature_values[feature_num]
            feature_names.append(feature)
            feature_name_to_num[feature] = feature_num
            feature_type, feature_internal_index = self._object._get_feature_type_and_internal_index(feature_num)
            if feature_type == 'categorical':
                cat_features_nums.append(feature_internal_index)
                feature_type_mapper.append('cat')
            else:
                float_features_nums.append(feature_internal_index)
                feature_type_mapper.append('float')
        results = [self._object._get_binarized_statistics(data_item, cat_features_nums, float_features_nums, prediction_type, thread_count) for data_item in data]
        statistics_by_feature = defaultdict(list)
        to_float_offset = len(cat_features_nums)
        cat_index, float_index = (0, to_float_offset)
        for i, type in enumerate(feature_type_mapper):
            feature_name = feature_names[i]
            feature_num = feature_name_to_num[feature_name]
            if type == 'cat':
                if feature_name not in cat_feature_values:
                    cat_feature_values_ = self._object._get_cat_feature_values(data[0], feature_num)
                    cat_feature_values_ = [val for val in cat_feature_values_]
                else:
                    cat_feature_values_ = cat_feature_values[feature_name]
                if not isinstance(cat_feature_values_, ARRAY_TYPES):
                    raise CatBoostError("Feature '{}' is categorical. Please provide values for which you need statistics in cat_feature_values".format(feature))
                val_to_hash = dict()
                for val in cat_feature_values_:
                    val_to_hash[val] = self._object._calc_cat_feature_perfect_hash(val, cat_features_nums[cat_index])
                hash_to_val = {hash: val for val, hash in val_to_hash.items()}
                for i, res in enumerate(results):
                    res[cat_index]['cat_values'] = np.array([hash_to_val[i] for i in sorted(hash_to_val.keys())])
                    res[cat_index].pop('borders', None)
                    statistics_by_feature[feature_num].append(res[cat_index])
                cat_index += 1
            else:
                for res in results:
                    statistics_by_feature[feature_num].append(res[float_index])
                float_index += 1
        if plot or plot_file is not None:
            fig = _plot_feature_statistics(statistics_by_feature, pool_names, self.feature_names_, max_cat_features_on_plot)
            if plot:
                try_plot_offline(fig)
            if plot_file is not None:
                save_plot_file(plot_file, 'Catboost metrics graph', [fig])
        for key in statistics_by_feature.keys():
            if len(statistics_by_feature[key]) == 1:
                statistics_by_feature[key] = statistics_by_feature[key][0]
        if is_for_one_feature:
            return statistics_by_feature[feature_name_to_num[feature_names[0]]]
        return_stats = {}
        for feature in features:
            if isinstance(feature, int):
                return_stats[feature] = statistics_by_feature[feature]
            else:
                return_stats[feature] = statistics_by_feature[feature_name_to_num[feature]]
        return return_stats

    def _plot_oblivious_tree(self, splits, leaf_values):
        from graphviz import Digraph
        graph = Digraph()
        layer_size = 1
        current_size = 0
        for split_num in range(len(splits) - 1, -2, -1):
            for node_num in range(layer_size):
                if split_num >= 0:
                    node_label = splits[split_num].replace('bin=', 'value>', 1).replace('border=', 'value>', 1)
                    color = 'black'
                    shape = 'ellipse'
                else:
                    node_label = leaf_values[node_num]
                    color = 'red'
                    shape = 'rect'
                try:
                    node_label = node_label.decode('utf-8')
                except Exception:
                    pass
                graph.node(str(current_size), node_label, color=color, shape=shape)
                if current_size > 0:
                    parent = (current_size - 1) // 2
                    edge_label = 'Yes' if current_size % 2 == 0 else 'No'
                    graph.edge(str(parent), str(current_size), edge_label)
                current_size += 1
            layer_size *= 2
        return graph

    def _plot_nonsymmetric_tree(self, splits, leaf_values, step_nodes, node_to_leaf):
        from graphviz import Digraph
        graph = Digraph()

        def plot_leaf(node_idx, graph):
            cur_id = 'leaf_{}'.format(node_to_leaf[node_idx])
            node_label = leaf_values[node_to_leaf[node_idx]]
            graph.node(cur_id, node_label, color='red', shape='rect')
            return cur_id

        def plot_subtree(node_idx, graph):
            if step_nodes[node_idx] == (0, 0):
                return plot_leaf(node_idx, graph)
            else:
                cur_id = 'node_{}'.format(node_idx)
                node_label = splits[node_idx].replace('bin=', 'value>', 1).replace('border=', 'value>', 1)
                graph.node(cur_id, node_label, color='black', shape='ellipse')
                if step_nodes[node_idx][0] == 0:
                    child_id = plot_leaf(node_idx, graph)
                else:
                    child_id = plot_subtree(node_idx + step_nodes[node_idx][0], graph)
                graph.edge(cur_id, child_id, 'No')
                if step_nodes[node_idx][1] == 0:
                    child_id = plot_leaf(node_idx, graph)
                else:
                    child_id = plot_subtree(node_idx + step_nodes[node_idx][1], graph)
                graph.edge(cur_id, child_id, 'Yes')
            return cur_id
        plot_subtree(0, graph)
        return graph

    def plot_tree(self, tree_idx, pool=None):
        pool, _ = self._process_predict_input_data(pool, 'plot_tree', thread_count=-1) if pool is not None else (None, None)
        splits = self._get_tree_splits(tree_idx, pool)
        leaf_values = self._get_tree_leaf_values(tree_idx)
        if self._object._is_oblivious():
            return self._plot_oblivious_tree(splits, leaf_values)
        else:
            step_nodes = self._get_tree_step_nodes(tree_idx)
            node_to_leaf = self._get_tree_node_to_leaf(tree_idx)
            return self._plot_nonsymmetric_tree(splits, leaf_values, step_nodes, node_to_leaf)

    def _tune_hyperparams(self, param_grid, X, y=None, cv=3, n_iter=10, partition_random_seed=0, calc_cv_statistics=True, search_by_train_test_split=True, refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=1, plot=False, plot_file=None, log_cout=None, log_cerr=None):
        if refit and self.is_fitted():
            raise CatBoostError("Model was fitted before hyperparameters tuning. You can't change hyperparameters of fitted model.")
        with log_fixup(log_cout, log_cerr):
            currently_not_supported_params = {'ignored_features', 'input_borders', 'loss_function', 'eval_metric'}
            if isinstance(param_grid, Mapping):
                param_grid = [param_grid]
            for grid_num, grid in enumerate(param_grid):
                _process_synonyms_groups(grid)
                grid = _params_type_cast(grid)
                for param in currently_not_supported_params:
                    if param in grid:
                        raise CatBoostError("Parameter '{}' currently is not supported in hyperparaneter search".format(param))
            if X is None:
                raise CatBoostError('X must not be None')
            if y is None and (not isinstance(X, PATH_TYPES + (Pool,))):
                raise CatBoostError('y may be None only when X is an instance of catboost.Pool, str or pathlib.Path')
            if not isinstance(param_grid, (Mapping, Iterable)):
                raise TypeError('Parameter grid is not a dict or a list ({!r})'.format(param_grid))
            train_params = self._prepare_train_params(X=X, y=y)
            params = train_params['params']
            custom_folds = None
            fold_count = 0
            if isinstance(cv, INTEGER_TYPES):
                fold_count = cv
                loss_function = params.get('loss_function', None)
                if stratified is None:
                    stratified = isinstance(loss_function, STRING_TYPES) and is_cv_stratified_objective(loss_function)
            else:
                if not hasattr(cv, '__iter__') and (not hasattr(cv, 'split')):
                    raise AttributeError('cv should be one of possible things:\n- None, to use the default 3-fold cross validation,\n- integer, to specify the number of folds in a (Stratified)KFold\n- one of the scikit-learn splitter classes (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)\n- An iterable yielding (train, test) splits as arrays of indices')
                custom_folds = cv
                shuffle = False
            if stratified is None:
                loss_function = params.get('loss_function', None)
                stratified = isinstance(loss_function, STRING_TYPES) and is_cv_stratified_objective(loss_function)
            with plot_wrapper(plot, plot_file, 'Hyperparameters search plot', [_get_train_dir(params)]):
                cv_result = self._object._tune_hyperparams(param_grid, train_params['train_pool'], params, n_iter, fold_count, partition_random_seed, shuffle, stratified, train_size, search_by_train_test_split, calc_cv_statistics, custom_folds, verbose)
            if refit:
                assert not self.is_fitted()
                self.set_params(**cv_result['params'])
                self.fit(X, y, silent=True)
        return cv_result

    def grid_search(self, param_grid, X, y=None, cv=3, partition_random_seed=0, calc_cv_statistics=True, search_by_train_test_split=True, refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False, plot_file=None, log_cout=None, log_cerr=None):
        """
        Exhaustive search over specified parameter values for a model.
        After calling this method model is fitted and can be used, if not specified otherwise (refit=False).

        Parameters
        ----------
        param_grid: dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of parameter settings
            to try as values, or a list of such dictionaries, in which case the grids spanned by each
            dictionary in the list are explored.
            This enables searching over any sequence of parameter settings.

        X: numpy.ndarray or pandas.DataFrame or catboost.Pool
            Data to compute statistics on

        y: list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels of the training data.
            If not None, can be a single- or two- dimensional array with either:
              - numerical values - for regression (including multiregression), ranking and binary classification problems
              - class labels (boolean, integer or string) - for classification (including multiclassification) problems
            Use only if X is not catboost.Pool and does not point to a file.

        cv: int, cross-validation generator or an iterable, optional (default=None)
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a (Stratified)KFold
            - one of the scikit-learn splitter classes
                (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
            - An iterable yielding (train, test) splits as arrays of indices.

        partition_random_seed: int, optional (default=0)
            Use this as the seed value for random permutation of the data.
            Permutation is performed before splitting the data for cross validation.
            Each seed generates unique data splits.
            Used only when cv is None or int.

        search_by_train_test_split: bool, optional (default=True)
            If True, source dataset is splitted into train and test parts, models are trained
            on the train part and parameters are compared by loss function score on the test part.
            After that, if calc_cv_statistics=true, statistics on metrics are calculated
            using cross-validation using best parameters and the model is fitted with these parameters.

            If False, every iteration of grid search evaluates results on cross-validation.
            It is recommended to set parameter to True for large datasets, and to False for small datasets.

        calc_cv_statistics: bool, optional (default=True)
            The parameter determines whether quality should be estimated.
            using cross-validation with the found best parameters. Used only when search_by_train_test_split=True.

        refit: bool (default=True)
            Refit an estimator using the best found parameters on the whole dataset.

        shuffle: bool, optional (default=True)
            Shuffle the dataset objects before parameters searching.

        stratified: bool, optional (default=None)
            Perform stratified sampling. True for classification and False otherwise.
            Currently supported only for final cross-validation.

        train_size: float, optional (default=0.8)
            Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.

        verbose: bool or int, optional (default=True)
            If verbose is int, it determines the frequency of writing metrics to output
            verbose==True is equal to verbose==1
            When verbose==False, there is no messages

        plot : bool, optional (default=False)
            If True, draw train and eval error for every set of parameters in Jupyter notebook

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error for every set of parameters to file

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        dict with two fields:
            'params': dict of best found parameters
            'cv_results': dict or pandas.core.frame.DataFrame with cross-validation results
                columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
        """
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        for grid in param_grid:
            if not isinstance(grid, Mapping):
                raise TypeError('Parameter grid is not a dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError('Parameter grid value is not iterable (key={!r}, value={!r})'.format(key, grid[key]))
        return self._tune_hyperparams(param_grid=param_grid, X=X, y=y, cv=cv, n_iter=-1, partition_random_seed=partition_random_seed, calc_cv_statistics=calc_cv_statistics, search_by_train_test_split=search_by_train_test_split, refit=refit, shuffle=shuffle, stratified=stratified, train_size=train_size, verbose=verbose, plot=plot, plot_file=plot_file, log_cout=log_cout, log_cerr=log_cerr)

    def randomized_search(self, param_distributions, X, y=None, cv=3, n_iter=10, partition_random_seed=0, calc_cv_statistics=True, search_by_train_test_split=True, refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False, plot_file=None, log_cout=None, log_cerr=None):
        """
        Randomized search on hyper parameters.
        After calling this method model is fitted and can be used, if not specified otherwise (refit=False).

        In contrast to grid_search, not all parameter values are tried out,
        but rather a fixed number of parameter settings is sampled from the specified distributions.
        The number of parameter settings that are tried is given by n_iter.

        Parameters
        ----------
        param_distributions: dict
            Dictionary with parameters names (string) as keys and distributions or lists of parameters to try.
            Distributions must provide a rvs method for sampling (such as those from scipy.stats.distributions).
            If a list is given, it is sampled uniformly.

        X: numpy.ndarray or pandas.DataFrame or catboost.Pool
            Data to compute statistics on

        y: list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels of the training data.
            If not None, can be a single- or two- dimensional array with either:
              - numerical values - for regression (including multiregression), ranking and binary classification problems
              - class labels (boolean, integer or string) - for classification (including multiclassification) problems
            Use only if X is not catboost.Pool and does not point to a file.

        cv: int, cross-validation generator or an iterable, optional (default=None)
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a (Stratified)KFold
            - one of the scikit-learn splitter classes
                (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
            - An iterable yielding (train, test) splits as arrays of indices.

        n_iter: int
            Number of parameter settings that are sampled.
            n_iter trades off runtime vs quality of the solution.

        partition_random_seed: int, optional (default=0)
            Use this as the seed value for random permutation of the data.
            Permutation is performed before splitting the data for cross validation.
            Each seed generates unique data splits.
            Used only when cv is None or int.

        search_by_train_test_split: bool, optional (default=True)
            If True, source dataset is splitted into train and test parts, models are trained
            on the train part and parameters are compared by loss function score on the test part.
            After that, if calc_cv_statistics=true, statistics on metrics are calculated
            using cross-validation using best parameters and the model is fitted with these parameters.

            If False, every iteration of grid search evaluates results on cross-validation.
            It is recommended to set parameter to True for large datasets, and to False for small datasets.

        calc_cv_statistics: bool, optional (default=True)
            The parameter determines whether quality should be estimated.
            using cross-validation with the found best parameters. Used only when search_by_train_test_split=True.

        refit: bool (default=True)
            Refit an estimator using the best found parameters on the whole dataset.

        shuffle: bool, optional (default=True)
            Shuffle the dataset objects before parameters searching.

        stratified: bool, optional (default=None)
            Perform stratified sampling. True for classification and False otherwise.
            Currently supported only for cross-validation.

        train_size: float, optional (default=0.8)
            Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.

        verbose: bool or int, optional (default=True)
            If verbose is int, it determines the frequency of writing metrics to output
            verbose==True is equal to verbose==1
            When verbose==False, there is no messages

        plot : bool, optional (default=False)
            If True, draw train and eval error for every set of parameters in Jupyter notebook

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error for every set of parameters to file

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        dict with two fields:
            'params': dict of best found parameters
            'cv_results': dict or pandas.core.frame.DataFrame with cross-validation results
                columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
        """
        if n_iter <= 0:
            assert CatBoostError('n_iter should be a positive number')
        if not isinstance(param_distributions, Mapping):
            assert CatBoostError('param_distributions should be a dictionary')
        for key in param_distributions:
            if not isinstance(param_distributions[key], Iterable) and (not hasattr(param_distributions[key], 'rvs')):
                raise TypeError("Parameter grid value is not iterable and do not have 'rvs' method (key={!r}, value={!r})".format(key, param_distributions[key]))
        return self._tune_hyperparams(param_grid=param_distributions, X=X, y=y, cv=cv, n_iter=n_iter, partition_random_seed=partition_random_seed, calc_cv_statistics=calc_cv_statistics, search_by_train_test_split=search_by_train_test_split, refit=refit, shuffle=shuffle, stratified=stratified, train_size=train_size, verbose=verbose, plot=plot, plot_file=plot_file, log_cout=log_cout, log_cerr=log_cerr)

    def select_features(self, X, y=None, eval_set=None, features_for_select=None, num_features_to_select=None, algorithm=None, steps=None, shap_calc_type=None, train_final_model=True, verbose=None, logging_level=None, plot=False, plot_file=None, log_cout=None, log_cerr=None, grouping=None, features_tags_for_select=None, num_features_tags_to_select=None):
        """
        Select best features from pool according to loss value.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels of the training data.
            If not None, can be a single- or two- dimensional array with either:
              - numerical values - for regression (including multiregression), ranking and binary classification problems
              - class labels (boolean, integer or string) - for classification (including multiclassification) problems
            Use only if X is not catboost.Pool and does not point to a file.

        eval_set : catboost.Pool or list of catboost.Pool or tuple (X, y) or list [(X, y)], optional (default=None)
            Validation dataset or datasets for metrics calculation and possibly early stopping.

        features_for_select : str or list of feature indices, names or ranges
            (for grouping = Individual)
            Which features should participate in the selection.
            Format examples:
                - [0, 2, 3, 4, 17]
                - [0, "2-4", 17] (both ends in ranges are inclusive)
                - "0,2-4,20"
                - ["Name0", "Name2", "Name3", "Name4", "Name20"]

        num_features_to_select : positive int
            (for grouping = Individual)
            How many features to select from features_for_select.

        algorithm : EFeaturesSelectionAlgorithm or string, optional (default=RecursiveByShapValues)
            Which algorithm to use for features selection.
            Possible values:
                - RecursiveByPredictionValuesChange
                    Use prediction values change as feature strength, eliminate batch of features at once.
                - RecursiveByLossFunctionChange
                    Use loss function change as feature strength, eliminate batch of features at each step.
                - RecursiveByShapValues
                    Use shap values to estimate loss function change, eliminate features one by one.

        steps : positive int, optional (default=1)
            How many steps should be performed. In other words, how many times a full model will be trained.
            More steps give more accurate results.

        shap_calc_type : EShapCalcType or string, optional (default=Regular)
            Which method to use for calculation of shap values.
            Possible values:
                - Regular
                    Calculate regular SHAP values
                - Approximate
                    Calculate approximate SHAP values
                - Exact
                    Calculate exact SHAP values

        train_final_model : bool, optional (default=True)
            Need to fit model with selected features.

        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook.

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error graphs to file

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        grouping : EFeaturesSelectionGrouping or string, optional (default=Individual)
            Which grouping to use for features selection.
            Possible values:
                - Individual
                    Select individual features
                - ByTags
                    Select feature groups (marked by tags)

        features_tags_for_select : list of strings
            (for grouping = ByTags)
            Which features tags should participate in the selection.

        num_features_tags_to_select : positive int
            (for grouping = ByTags)
            How many features tags to select from features_tags_for_select.

        Returns
        -------
        dict with fields:
            'selected_features': list of selected features indices
            'eliminated_features': list of eliminated features indices
            'selected_features_tags': list of selected features tags (optional, present if grouping == ByTags)
            'eliminated_features_tags': list of selected features tags (optional, present if grouping == ByTags)
        """
        if train_final_model and self.is_fitted():
            raise CatBoostError('Model was already fitted. Set train_final_model to False or use not fitted model.')
        if X is None:
            raise CatBoostError('X must not be None')
        if y is None and (not isinstance(X, PATH_TYPES + (Pool,))):
            raise CatBoostError('y may be None only when X is an instance of catboost.Pool, str or pathlib.Path.')
        with log_fixup(log_cout, log_cerr):
            train_params = self._prepare_train_params(X=X, y=y, eval_set=eval_set, verbose=verbose, logging_level=logging_level)
            params = train_params['params']
            if grouping is None:
                grouping = EFeaturesSelectionGrouping.Individual
            else:
                grouping = enum_from_enum_or_str(EFeaturesSelectionGrouping, grouping).value
                params['features_selection_grouping'] = grouping
            if grouping == EFeaturesSelectionGrouping.Individual:
                if isinstance(features_for_select, Iterable) and (not isinstance(features_for_select, STRING_TYPES)):
                    features_for_select = ','.join(map(str, features_for_select))
                if features_for_select is None:
                    raise CatBoostError('You should specify features_for_select')
                if features_tags_for_select is not None:
                    raise CatBoostError('You should not specify features_tags_for_select when grouping is Individual')
                if num_features_to_select is None:
                    raise CatBoostError('You should specify num_features_to_select')
                if num_features_tags_to_select is not None:
                    raise CatBoostError('You should not specify num_features_tags_to_select when grouping is Individual')
                params['features_for_select'] = features_for_select
                params['num_features_to_select'] = num_features_to_select
            else:
                if features_tags_for_select is None:
                    raise CatBoostError('You should specify features_tags_for_select')
                if not isinstance(features_tags_for_select, Sequence):
                    raise CatBoostError('features_tags_for_select must be a list of strings')
                if features_for_select is not None:
                    raise CatBoostError('You should not specify features_for_select when grouping is ByTags')
                if num_features_tags_to_select is None:
                    raise CatBoostError('You should specify num_features_tags_to_select')
                if num_features_to_select is not None:
                    raise CatBoostError('You should not specify num_features_to_select when grouping is ByTags')
                params['features_tags_for_select'] = features_tags_for_select
                params['num_features_tags_to_select'] = num_features_tags_to_select
            objective = params.get('loss_function')
            is_custom_objective = objective is not None and (not isinstance(objective, string_types))
            if is_custom_objective:
                raise CatBoostError('Custom objective is not supported for features selection')
            if algorithm is not None:
                params['features_selection_algorithm'] = enum_from_enum_or_str(EFeaturesSelectionAlgorithm, algorithm).value
            if steps is not None:
                params['features_selection_steps'] = steps
            if shap_calc_type is not None:
                params['shap_calc_type'] = enum_from_enum_or_str(EShapCalcType, shap_calc_type).value
            if train_final_model:
                params['train_final_model'] = True
            train_pool = train_params['train_pool']
            test_pool = None
            if len(train_params['eval_sets']) > 1:
                raise CatBoostError('Multiple eval sets are not supported for features selection')
            elif len(train_params['eval_sets']) == 1:
                test_pool = train_params['eval_sets'][0]
            train_dir = _get_train_dir(self.get_params())
            create_dir_if_not_exist(train_dir)
            plot_dirs = []
            for step in range(steps or 1):
                plot_dirs.append(os.path.join(train_dir, 'model-{}'.format(step)))
            if train_final_model:
                plot_dirs.append(os.path.join(train_dir, 'model-final'))
            for plot_dir in plot_dirs:
                create_dir_if_not_exist(plot_dir)
            with plot_wrapper(plot, plot_file=plot_file, plot_title='Select features plot', train_dirs=plot_dirs):
                summary = self._object._select_features(train_pool, test_pool, params)
            if train_final_model:
                self._set_trained_model_attributes()
            if plot:
                figures = plot_features_selection_loss_graphs(summary)
                figures['features'].show()
                if 'features_tags' in figures:
                    figures['features_tags'].show()
        return summary

    def _convert_to_asymmetric_representation(self):
        self._object._convert_oblivious_to_asymmetric()