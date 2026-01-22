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
class CatBoostRanker(CatBoost):
    """
    Implementation of the scikit-learn API for CatBoost ranking.
    Parameters
    ----------
    Like in CatBoostClassifier, except loss_function, classes_count, class_names and class_weights
    loss_function : string, [default='YetiRank']
        'YetiRank'
        'YetiRankPairwise'
        'StochasticFilter'
        'StochasticRank'
        'QueryCrossEntropy'
        'QueryRMSE'
        'QuerySoftMax'
        'PairLogit'
        'PairLogitPairwise'
    """
    _estimator_type = 'ranker'

    def __init__(self, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None, model_size_reg=None, rsm=None, loss_function='YetiRank', border_count=None, feature_border_type=None, per_float_feature_quantization=None, input_borders=None, output_borders=None, fold_permutation_block=None, od_pval=None, od_wait=None, od_type=None, nan_mode=None, counter_calc_method=None, leaf_estimation_iterations=None, leaf_estimation_method=None, thread_count=None, random_seed=None, use_best_model=None, best_model_min_trees=None, verbose=None, silent=None, logging_level=None, metric_period=None, ctr_leaf_count_limit=None, store_all_simple_ctr=None, max_ctr_complexity=None, has_time=None, allow_const_label=None, target_border=None, one_hot_max_size=None, random_strength=None, random_score_type=None, name=None, ignored_features=None, train_dir=None, custom_metric=None, eval_metric=None, bagging_temperature=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, fold_len_multiplier=None, used_ram_limit=None, gpu_ram_part=None, pinned_memory_size=None, allow_writing_files=None, final_ctr_computation_mode=None, approx_on_full_history=None, boosting_type=None, simple_ctr=None, combinations_ctr=None, per_feature_ctr=None, ctr_description=None, ctr_target_border_count=None, task_type=None, device_config=None, devices=None, bootstrap_type=None, subsample=None, mvs_reg=None, sampling_frequency=None, sampling_unit=None, dev_score_calc_obj_block_size=None, dev_efb_max_buckets=None, sparse_features_conflict_fraction=None, max_depth=None, n_estimators=None, num_boost_round=None, num_trees=None, colsample_bylevel=None, random_state=None, reg_lambda=None, objective=None, eta=None, max_bin=None, gpu_cat_features_storage=None, data_partition=None, metadata=None, early_stopping_rounds=None, cat_features=None, grow_policy=None, min_data_in_leaf=None, min_child_samples=None, max_leaves=None, num_leaves=None, score_function=None, leaf_estimation_backtracking=None, ctr_history_unit=None, monotone_constraints=None, feature_weights=None, penalties_coefficient=None, first_feature_use_penalties=None, per_object_feature_penalties=None, model_shrink_rate=None, model_shrink_mode=None, langevin=None, diffusion_temperature=None, posterior_sampling=None, boost_from_average=None, text_features=None, tokenizers=None, dictionaries=None, feature_calcers=None, text_processing=None, embedding_features=None, eval_fraction=None, fixed_binary_splits=None):
        params = {}
        not_params = ['not_params', 'self', 'params', '__class__']
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value
        super(CatBoostRanker, self).__init__(params)

    def fit(self, X, y=None, group_id=None, cat_features=None, text_features=None, embedding_features=None, pairs=None, sample_weight=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False, plot_file=None, column_description=None, verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None, log_cout=None, log_cerr=None):
        """
        Fit the CatBoostRanker model.
        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.
        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels of the training data.
            If not None, can be a single-dimensional array with numerical values.
            Use only if X is not catboost.Pool and does not point to a file.
        group_id : numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Ranking groups, 1 dimensional array like.
            Use only if X is not catboost.Pool.
        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool.
        text_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Text columns indices.
            Use only if X is not catboost.Pool.
        embedding_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Embedding columns indices.
            Use only if X is not catboost.Pool.
        pairs : list or numpy.ndarray or pandas.DataFrame, optional (default=None)
            The pairs description in the form of a two-dimensional matrix of shape N by 2:
            N is the number of pairs.
            The first element of the pair is the zero-based index of the winner object from the input dataset for pairwise comparison.
            The second element of the pair is the zero-based index of the loser object from the input dataset for pairwise comparison.
        sample_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.
        group_weight : list or numpy.ndarray (default=None)
            The weights of all objects within the defined groups from the input data in the form of one-dimensional array-like data.
            Used for calculating the final values of trees. By default, it is set to 1 for all objects in all groups.
            Only a weight or group_weight parameter can be used at a time
        subgroup_id : list or numpy.ndarray (default=None)
            Subgroup identifiers for all input objects. Supported identifier types are:
            int
            string types (string or unicode for Python 2 and bytes or string for Python 3).
        pairs_weight : list or numpy.ndarray (default=None)
            The weight of each input pair of objects in the form of one-dimensional array-like pairs.
            The number of given values must match the number of specified pairs.
            This information is used for calculation and optimization of Pairwise metrics .
            By default, it is set to 1 for all pairs.
        baseline : list or numpy.ndarray, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not catboost.Pool.
        use_best_model : bool, optional (default=None)
            Flag to use best model
        eval_set : catboost.Pool or list of catboost.Pool or tuple (X, y) or list [(X, y)], optional (default=None)
            Validation dataset or datasets for metrics calculation and possibly early stopping.
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
            If True, draw train and eval error in Jupyter notebook
        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error graphs to file
        verbose_eval : bool or int
            Synonym for verbose. Only one of these parameters should be set.
        metric_period : int
            Frequency of evaluating metrics.
        silent : bool
            If silent is True, logging_level is set to Silent.
            If silent is False, logging_level is set to Verbose.
        early_stopping_rounds : int
            Activates Iter overfitting detector with od_wait set to early_stopping_rounds.
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
        params = deepcopy(self._init_params)
        _process_synonyms(params)
        if 'loss_function' in params:
            CatBoostRanker._check_is_compatible_loss(params['loss_function'])
        self._fit(X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
        return self

    def predict(self, X, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict with data.
        Parameters
        ----------
        X : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.
        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.
        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.
        verbose : bool
            If True, writes the evaluation metric measured set to stderr.
        Returns
        -------
        prediction :
            If data is for a single object, the return value is single float formula return value
            otherwise one-dimensional numpy.ndarray of formula return values for each object.
        """
        return self._predict(X, 'RawFormulaVal', ntree_start, ntree_end, thread_count, verbose, 'predict')

    def staged_predict(self, X, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.
        Parameters
        ----------
        X : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.
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
            If data is for a single object, the return value is single float formula return value
            otherwise one-dimensional numpy.ndarray of formula return values for each object.
        """
        return self._staged_predict(X, 'RawFormulaVal', ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def score(self, X, y=None, group_id=None, top=None, type=None, denominator=None, group_weight=None, thread_count=-1):
        """
        Calculate NDCG@top
        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Data to apply model on.
        y : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
            True labels.
        group_id : list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Ranking groups. If X is a Pool, group_id must be defined into X
        top : unsigned integer, up to `pow(2, 32) / 2 - 1`
            NDCG, Number of top-ranked objects to calculate NDCG
        type : str
            NDCG, Metric_type: 'Base' or 'Exp'
        denominator : str
            NDCG, Denominator type: 'LogPosition' or 'Position'
        group_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Group weights.
        thread_count : int, optional (default=-1)
            Number of threads to work with.
        Returns
        -------
        NDCG@top : float
                   higher is better
        """

        def get_ndcg_metric_name(values, names):
            if np.all(np.equal(values, None)):
                return 'NDCG'
            return 'NDCG:' + ';'.join(['{}={}'.format(n, v) for v, n in zip(values, names) if v is not None])
        if isinstance(X, Pool):
            if y is not None:
                raise CatBoostError('Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool.')
            y = X.get_label()
            if group_id is not None:
                raise CatBoostError('Wrong initializing group_id: X is catboost.Pool object, group_id must be initialized inside catboost.Pool.')
            group_id = X.get_group_id_hash()
        if y is None:
            raise CatBoostError('y must be initialized.')
        if group_id is None:
            raise CatBoostError('group_id must be initialized. If groups are not expected, pass an array of zeros')
        predictions = self.predict(X)
        return _eval_metric_util([y], [predictions], get_ndcg_metric_name([top, type, denominator], ['top', 'type', 'denominator']), None, group_id, group_weight, None, None, thread_count)[0]

    @staticmethod
    def _check_is_compatible_loss(loss_function):
        is_ranking = CatBoost._is_ranking_objective(loss_function)
        is_regression = CatBoost._is_regression_objective(loss_function)
        if is_regression:
            warnings.warn("Regression loss ('{}') ignores an important ranking parameter 'group_id'".format(loss_function), RuntimeWarning)
        if not (is_ranking or is_regression):
            raise CatBoostError("Invalid loss_function='{}': for ranker use YetiRank, YetiRankPairwise, StochasticFilter, StochasticRank, QueryCrossEntropy, QueryRMSE, QuerySoftMax, PairLogit, PairLogitPairwise. It's also possible to use a regression loss".format(loss_function))