import logging
import os
import tempfile
import warnings
from packaging.version import Version
import mlflow.pytorch
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.pytorch import _pytorch_autolog
from mlflow.utils.autologging_utils import (
from mlflow.utils.checkpoint_utils import MlflowModelCheckpointCallbackBase
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

            Log loss and other metrics values after each epoch

            Args:
                trainer: pytorch lightning trainer instance
                pl_module: pytorch lightning base module
            