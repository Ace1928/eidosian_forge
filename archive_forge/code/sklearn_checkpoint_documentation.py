import os
import tempfile
from typing import TYPE_CHECKING, Optional, Union
from sklearn.base import BaseEstimator
import ray.cloudpickle as cpickle
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
Retrieve the ``Estimator`` stored in this checkpoint.