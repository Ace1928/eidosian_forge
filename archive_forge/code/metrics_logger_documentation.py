import sys
from typing import Any, Dict, Optional, Union
import tensorflow as tf  # type: ignore
from tensorflow.keras import callbacks  # type: ignore
import wandb
from wandb.integration.keras.keras import patch_tf_keras
from wandb.sdk.lib import telemetry
Called at the end of a training batch in `fit` methods.