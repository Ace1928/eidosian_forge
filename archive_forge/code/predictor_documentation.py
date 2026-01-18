import abc
from typing import Callable, Dict, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import (
from ray.data import Preprocessor
from ray.train import Checkpoint
from ray.util.annotations import DeveloperAPI, PublicAPI
Perform inference on a Numpy data.

        All Predictors working with tensor data (like deep learning predictors)
        should implement this method.

        Args:
            data: A Numpy ndarray or dictionary of ndarrays to perform predictions on.
            kwargs: Arguments specific to the predictor implementation.

        Returns:
            A Numpy ndarray or dictionary of ndarray containing the prediction result.

        