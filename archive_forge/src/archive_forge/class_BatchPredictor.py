import logging
from typing import Type
from ray.train import Checkpoint
from ray.train.predictor import Predictor
from ray.util.annotations import Deprecated
@Deprecated(message=BATCH_PREDICTION_DEPRECATION_MSG)
class BatchPredictor:
    """Batch predictor class.

    Takes a predictor class and a checkpoint and provides an interface to run
    batch scoring on Datasets.

    This batch predictor wraps around a predictor class and executes it
    in a distributed way when calling ``predict()``.
    """

    def __init__(self, checkpoint: Checkpoint, predictor_cls: Type[Predictor], **predictor_kwargs):
        raise DeprecationWarning(BATCH_PREDICTION_DEPRECATION_MSG)