import os
import ray
from ray.air.constants import COPY_DIRECTORY_CHECKPOINTS_INSTEAD_OF_MOVING_ENV
from ray.train.constants import RAY_CHDIR_TO_TRIAL_DIR
Gets the wrapped trainable_cls, otherwise calls ray.remote.