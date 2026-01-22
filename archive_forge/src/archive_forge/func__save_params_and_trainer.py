import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
def _save_params_and_trainer(self, estimator, file_prefix):
    param_file = os.path.join(self.model_dir, file_prefix + '.params')
    trainer_file = os.path.join(self.model_dir, file_prefix + '.states')
    estimator.net.save_parameters(param_file)
    estimator.trainer.save_states(trainer_file)
    if 'best' not in file_prefix:
        self.saved_checkpoints.append(file_prefix)
    if len(self.saved_checkpoints) > self.max_checkpoints:
        prefix = self.saved_checkpoints.pop(0)
        for fname in os.listdir(self.model_dir):
            if fname.startswith(prefix):
                os.remove(os.path.join(self.model_dir, fname))