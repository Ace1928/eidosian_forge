from typing import TYPE_CHECKING, List, Optional, Union
import lightgbm
import pandas as pd
from pandas.api.types import is_object_dtype
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
from ray.train.lightgbm import LightGBMCheckpoint
from ray.train.predictor import Predictor
from ray.util.annotations import PublicAPI
Run inference on data batch.

        Args:
            data: A batch of input data.
            feature_columns: The names or indices of the columns in the
                data to use as features to predict on. If None, then use
                all columns in ``data``.
            **predict_kwargs: Keyword arguments passed to
                ``lightgbm.Booster.predict``.

        Examples:
            >>> import numpy as np
            >>> import lightgbm as lgbm
            >>> from ray.train.lightgbm import LightGBMPredictor
            >>>
            >>> train_X = np.array([[1, 2], [3, 4]])
            >>> train_y = np.array([0, 1])
            >>>
            >>> model = lgbm.LGBMClassifier().fit(train_X, train_y)
            >>> predictor = LightGBMPredictor(model=model.booster_)
            >>>
            >>> data = np.array([[1, 2], [3, 4]])
            >>> predictions = predictor.predict(data)
            >>>
            >>> # Only use first and second column as the feature
            >>> data = np.array([[1, 2, 8], [3, 4, 9]])
            >>> predictions = predictor.predict(data, feature_columns=[0, 1])

            >>> import pandas as pd
            >>> import lightgbm as lgbm
            >>> from ray.train.lightgbm import LightGBMPredictor
            >>>
            >>> train_X = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
            >>> train_y = pd.Series([0, 1])
            >>>
            >>> model = lgbm.LGBMClassifier().fit(train_X, train_y)
            >>> predictor = LightGBMPredictor(model=model.booster_)
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

        