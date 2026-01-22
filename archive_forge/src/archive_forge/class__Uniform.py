import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
class _Uniform(Uniform):

    def sample(self, domain: 'Categorical', config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
        if not isinstance(random_state, _BackwardsCompatibleNumpyRng):
            random_state = _BackwardsCompatibleNumpyRng(random_state)
        indices = random_state.choice(np.arange(0, len(domain.categories)), size=size)
        items = [domain.categories[index] for index in indices]
        return items if len(items) > 1 else domain.cast(items[0])