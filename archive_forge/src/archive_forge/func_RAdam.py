import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, cast
from .backends import get_array_ops
from .config import registry
from .types import FloatsXd, Generator
@registry.optimizers('RAdam.v1')
def RAdam(learn_rate: FloatOrSeq=ADAM_DEFAULTS['learn_rate'], *, beta1: FloatOrSeq=ADAM_DEFAULTS['beta1'], beta2: FloatOrSeq=ADAM_DEFAULTS['beta2'], eps: FloatOrSeq=ADAM_DEFAULTS['eps'], L2: FloatOrSeq=ADAM_DEFAULTS['L2'], L2_is_weight_decay: bool=cast(bool, ADAM_DEFAULTS['L2_is_weight_decay']), grad_clip: FloatOrSeq=ADAM_DEFAULTS['grad_clip'], use_averages: bool=True):
    return Optimizer(learn_rate, beta1=beta1, beta2=beta2, eps=eps, grad_clip=grad_clip, L2_is_weight_decay=L2_is_weight_decay, L2=L2, use_averages=use_averages, use_radam=True)