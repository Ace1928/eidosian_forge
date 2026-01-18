from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import pandas as pd
import xgboost
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
from ray.train.predictor import Predictor
from ray.train.xgboost import XGBoostCheckpoint
from ray.util.annotations import PublicAPI
Run inference on data batch.

        The data is converted into an XGBoost DMatrix before being inputted to
        the model.

        Args:
            data: A batch of input data.
            feature_columns: The names or indices of the columns in the
                data to use as features to predict on. If None, then use
                all columns in ``data``.
            dmatrix_kwargs: Dict of keyword arguments passed to ``xgboost.DMatrix``.
            **predict_kwargs: Keyword arguments passed to ``xgboost.Booster.predict``.


        Examples:

        .. testcode::

            import numpy as np
            import xgboost as xgb
            from ray.train.xgboost import XGBoostPredictor
            train_X = np.array([[1, 2], [3, 4]])
            train_y = np.array([0, 1])
            model = xgb.XGBClassifier().fit(train_X, train_y)
            predictor = XGBoostPredictor(model=model.get_booster())
            data = np.array([[1, 2], [3, 4]])
            predictions = predictor.predict(data)
            # Only use first and second column as the feature
            data = np.array([[1, 2, 8], [3, 4, 9]])
            predictions = predictor.predict(data, feature_columns=[0, 1])

        .. testcode::

            import pandas as pd
            import xgboost as xgb
            from ray.train.xgboost import XGBoostPredictor
            train_X = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
            train_y = pd.Series([0, 1])
            model = xgb.XGBClassifier().fit(train_X, train_y)
            predictor = XGBoostPredictor(model=model.get_booster())
            # Pandas dataframe.
            data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
            predictions = predictor.predict(data)
            # Only use first and second column as the feature
            data = pd.DataFrame([[1, 2, 8], [3, 4, 9]], columns=["A", "B", "C"])
            predictions = predictor.predict(data, feature_columns=["A", "B"])


        Returns:
            Prediction result.

        