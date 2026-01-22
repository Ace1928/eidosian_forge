import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class Grid(Sampler):
    """Dummy sampler used for grid search"""

    def sample(self, domain: Domain, config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
        return RuntimeError('Do not call `sample()` on grid.')