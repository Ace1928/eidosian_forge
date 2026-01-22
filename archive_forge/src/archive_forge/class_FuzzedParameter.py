import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
class FuzzedParameter:
    """Specification for a parameter to be generated during fuzzing."""

    def __init__(self, name: str, minval: Optional[Union[int, float]]=None, maxval: Optional[Union[int, float]]=None, distribution: Optional[Union[str, Dict[Any, float]]]=None, strict: bool=False):
        """
        Args:
            name:
                A string name with which to identify the parameter.
                FuzzedTensors can reference this string in their
                specifications.
            minval:
                The lower bound for the generated value. See the description
                of `distribution` for type behavior.
            maxval:
                The upper bound for the generated value. Type behavior is
                identical to `minval`.
            distribution:
                Specifies the distribution from which this parameter should
                be drawn. There are three possibilities:
                    - "loguniform"
                        Samples between `minval` and `maxval` (inclusive) such
                        that the probabilities are uniform in log space. As a
                        concrete example, if minval=1 and maxval=100, a sample
                        is as likely to fall in [1, 10) as it is [10, 100].
                    - "uniform"
                        Samples are chosen with uniform probability between
                        `minval` and `maxval` (inclusive). If either `minval`
                        or `maxval` is a float then the distribution is the
                        continuous uniform distribution; otherwise samples
                        are constrained to the integers.
                    - dict:
                        If a dict is passed, the keys are taken to be choices
                        for the variables and the values are interpreted as
                        probabilities. (And must sum to one.)
                If a dict is passed, `minval` and `maxval` must not be set.
                Otherwise, they must be set.
            strict:
                If a parameter is strict, it will not be included in the
                iterative resampling process which Fuzzer uses to find a
                valid parameter configuration. This allows an author to
                prevent skew from resampling for a given parameter (for
                instance, a low size limit could inadvertently bias towards
                Tensors with fewer dimensions) at the cost of more iterations
                when generating parameters.
        """
        self._name = name
        self._minval = minval
        self._maxval = maxval
        self._distribution = self._check_distribution(distribution)
        self.strict = strict

    @property
    def name(self):
        return self._name

    def sample(self, state):
        if self._distribution == 'loguniform':
            return self._loguniform(state)
        if self._distribution == 'uniform':
            return self._uniform(state)
        if isinstance(self._distribution, dict):
            return self._custom_distribution(state)

    def _check_distribution(self, distribution):
        if not isinstance(distribution, dict):
            assert distribution in _DISTRIBUTIONS
        else:
            assert not any((i < 0 for i in distribution.values())), 'Probabilities cannot be negative'
            assert abs(sum(distribution.values()) - 1) <= 1e-05, 'Distribution is not normalized'
            assert self._minval is None
            assert self._maxval is None
        return distribution

    def _loguniform(self, state):
        output = int(2 ** state.uniform(low=np.log2(self._minval) if self._minval is not None else None, high=np.log2(self._maxval) if self._maxval is not None else None))
        if self._minval is not None and output < self._minval:
            return self._minval
        if self._maxval is not None and output > self._maxval:
            return self._maxval
        return output

    def _uniform(self, state):
        if isinstance(self._minval, int) and isinstance(self._maxval, int):
            return int(state.randint(low=self._minval, high=self._maxval + 1))
        return state.uniform(low=self._minval, high=self._maxval)

    def _custom_distribution(self, state):
        index = state.choice(np.arange(len(self._distribution)), p=tuple(self._distribution.values()))
        return list(self._distribution.keys())[index]