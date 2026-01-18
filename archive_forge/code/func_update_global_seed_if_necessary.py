import numpy as np
import os
import random
from typing import Optional
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_tf, try_import_torch
@DeveloperAPI
def update_global_seed_if_necessary(framework: Optional[str]=None, seed: Optional[int]=None) -> None:
    """Seed global modules such as random, numpy, torch, or tf.

    This is useful for debugging and testing.

    Args:
        framework: The framework specifier (may be None).
        seed: An optional int seed. If None, will not do
            anything.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if framework == 'torch':
        torch, _ = try_import_torch()
        torch.manual_seed(seed)
        cuda_version = torch.version.cuda
        if cuda_version is not None and float(torch.version.cuda) >= 10.2:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
        else:
            from packaging.version import Version
            if Version(torch.__version__) >= Version('1.8.0'):
                torch.use_deterministic_algorithms(True)
            else:
                torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
    elif framework == 'tf2':
        tf1, tf, tfv = try_import_tf()
        if tfv == 2:
            tf.random.set_seed(seed)
        else:
            tf1.set_random_seed(seed)