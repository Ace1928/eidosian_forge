from typing import TYPE_CHECKING, List, Optional, Union
import pandas as pd
from joblib import parallel_backend
from sklearn.base import BaseEstimator
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
from ray.train.predictor import Predictor
from ray.train.sklearn import SklearnCheckpoint
from ray.train.sklearn._sklearn_utils import _set_cpu_params
from ray.util.annotations import PublicAPI
from ray.util.joblib import register_ray
Run inference on data batch.

        Args:
            data: A batch of input data. Either a pandas DataFrame or numpy
                array.
            feature_columns: The names or indices of the columns in the
                data to use as features to predict on. If None, then use
                all columns in ``data``.
            num_estimator_cpus: If set to a value other than None, will set
                the values of all ``n_jobs`` and ``thread_count`` parameters
                in the estimator (including in nested objects) to the given value.
            **predict_kwargs: Keyword arguments passed to ``estimator.predict``.

        Examples:
            >>> import numpy as np
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from ray.train.sklearn import SklearnPredictor
            >>>
            >>> train_X = np.array([[1, 2], [3, 4]])
            >>> train_y = np.array([0, 1])
            >>>
            >>> model = RandomForestClassifier().fit(train_X, train_y)
            >>> predictor = SklearnPredictor(estimator=model)
            >>>
            >>> data = np.array([[1, 2], [3, 4]])
            >>> predictions = predictor.predict(data)
            >>>
            >>> # Only use first and second column as the feature
            >>> data = np.array([[1, 2, 8], [3, 4, 9]])
            >>> predictions = predictor.predict(data, feature_columns=[0, 1])

            >>> import pandas as pd
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from ray.train.sklearn import SklearnPredictor
            >>>
            >>> train_X = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
            >>> train_y = pd.Series([0, 1])
            >>>
            >>> model = RandomForestClassifier().fit(train_X, train_y)
            >>> predictor = SklearnPredictor(estimator=model)
            >>>
            >>> # Pandas dataframe.
            >>> data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
            >>> predictions = predictor.predict(data)
            >>>
            >>> # Only use first and second column as the feature
            >>> data = pd.DataFrame([[1, 2, 8], [3, 4, 9]], columns=["A", "B", "C"])
            >>> predictions = predictor.predict(data, feature_columns=["A", "B"])


        Returns:
            Prediction result.

        