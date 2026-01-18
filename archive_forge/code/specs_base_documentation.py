import abc
from copy import deepcopy
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union, Type
from ray.rllib.utils import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import TensorType
Checks if the shape and dtype of two specs are equal.