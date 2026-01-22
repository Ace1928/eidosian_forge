import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
def _can_compute_flops() -> bool:
    """FLOPS computation is restricted to TF 2.x as it requires tf.compat.v1."""
    from wandb.util import parse_version
    if parse_version(tf.__version__) >= parse_version('2.0.0'):
        return True
    return False