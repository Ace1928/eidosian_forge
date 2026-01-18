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