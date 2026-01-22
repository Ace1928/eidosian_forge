import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
class _CallSampler(BaseSampler):

    def __try_fn(self, domain: 'Function', config: Dict[str, Any]):
        try:
            return domain.func(config)
        except (AttributeError, KeyError):
            from ray.tune.search.variant_generator import _UnresolvedAccessGuard
            r = domain.func(_UnresolvedAccessGuard({'config': config}))
            logger.warning('sample_from functions that take a spec dict are deprecated. Please update your function to work with the config dict directly.')
            return r

    def sample(self, domain: 'Function', config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
        if not isinstance(random_state, _BackwardsCompatibleNumpyRng):
            random_state = _BackwardsCompatibleNumpyRng(random_state)
        if domain.pass_config:
            items = [self.__try_fn(domain, config[i]) if isinstance(config, list) else self.__try_fn(domain, config) for i in range(size)]
        else:
            items = [domain.func() for i in range(size)]
        return items if len(items) > 1 else domain.cast(items[0])