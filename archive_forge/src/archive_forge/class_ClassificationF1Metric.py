from parlai.core.opt import Opt
from parlai.utils.torch import PipelineHelper
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.core.metrics import Metric, AverageMetric
from typing import List, Optional, Tuple, Dict
from parlai.utils.typing import TScalar
import parlai.utils.logging as logging
import torch
import torch.nn.functional as F
class ClassificationF1Metric(ConfusionMatrixMetric):
    """
    Class that takes in a ConfusionMatrixMetric and computes f1 for classifier.
    """

    def value(self) -> float:
        if self._true_positives == 0:
            return 0.0
        else:
            numer = 2 * self._true_positives
            denom = numer + self._false_negatives + self._false_positives
            return numer / denom