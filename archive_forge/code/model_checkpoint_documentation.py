import os
import string
import sys
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf  # type: ignore
from tensorflow.keras import callbacks  # type: ignore
import wandb
from wandb.sdk.lib import telemetry
from wandb.sdk.lib.paths import StrPath
from ..keras import patch_tf_keras
Log model checkpoint as  W&B Artifact.