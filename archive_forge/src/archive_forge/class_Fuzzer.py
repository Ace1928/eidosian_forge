import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
class Fuzzer:

    def __init__(self, parameters: List[Union[FuzzedParameter, List[FuzzedParameter]]], tensors: List[Union[FuzzedTensor, List[FuzzedTensor]]], constraints: Optional[List[Callable]]=None, seed: Optional[int]=None):
        """
        Args:
            parameters:
                List of FuzzedParameters which provide specifications
                for generated parameters. Iterable elements will be
                unpacked, though arbitrary nested structures will not.
            tensors:
                List of FuzzedTensors which define the Tensors which
                will be created each step based on the parameters for
                that step. Iterable elements will be unpacked, though
                arbitrary nested structures will not.
            constraints:
                List of callables. They will be called with params
                as kwargs, and if any of them return False the current
                set of parameters will be rejected.
            seed:
                Seed for the RandomState used by the Fuzzer. This will
                also be used to set the PyTorch random seed so that random
                ops will create reproducible Tensors.
        """
        if seed is None:
            seed = np.random.RandomState().randint(0, 2 ** 32 - 1, dtype=np.int64)
        self._seed = seed
        self._parameters = Fuzzer._unpack(parameters, FuzzedParameter)
        self._tensors = Fuzzer._unpack(tensors, FuzzedTensor)
        self._constraints = constraints or ()
        p_names = {p.name for p in self._parameters}
        t_names = {t.name for t in self._tensors}
        name_overlap = p_names.intersection(t_names)
        if name_overlap:
            raise ValueError(f'Duplicate names in parameters and tensors: {name_overlap}')
        self._rejections = 0
        self._total_generated = 0

    @staticmethod
    def _unpack(values, cls):
        return tuple(it.chain(*[[i] if isinstance(i, cls) else i for i in values]))

    def take(self, n):
        state = np.random.RandomState(self._seed)
        torch.manual_seed(state.randint(low=0, high=2 ** 63, dtype=np.int64))
        for _ in range(n):
            params = self._generate(state)
            tensors = {}
            tensor_properties = {}
            for t in self._tensors:
                tensor, properties = t._make_tensor(params, state)
                tensors[t.name] = tensor
                tensor_properties[t.name] = properties
            yield (tensors, tensor_properties, params)

    @property
    def rejection_rate(self):
        if not self._total_generated:
            return 0.0
        return self._rejections / self._total_generated

    def _generate(self, state):
        strict_params: Dict[str, Union[float, int, ParameterAlias]] = {}
        for _ in range(1000):
            candidate_params: Dict[str, Union[float, int, ParameterAlias]] = {}
            for p in self._parameters:
                if p.strict:
                    if p.name in strict_params:
                        candidate_params[p.name] = strict_params[p.name]
                    else:
                        candidate_params[p.name] = p.sample(state)
                        strict_params[p.name] = candidate_params[p.name]
                else:
                    candidate_params[p.name] = p.sample(state)
            candidate_params = self._resolve_aliases(candidate_params)
            self._total_generated += 1
            if not all((f(candidate_params) for f in self._constraints)):
                self._rejections += 1
                continue
            if not all((t.satisfies_constraints(candidate_params) for t in self._tensors)):
                self._rejections += 1
                continue
            return candidate_params
        raise ValueError('Failed to generate a set of valid parameters.')

    @staticmethod
    def _resolve_aliases(params):
        params = dict(params)
        alias_count = sum((isinstance(v, ParameterAlias) for v in params.values()))
        keys = list(params.keys())
        while alias_count:
            for k in keys:
                v = params[k]
                if isinstance(v, ParameterAlias):
                    params[k] = params[v.alias_to]
            alias_count_new = sum((isinstance(v, ParameterAlias) for v in params.values()))
            if alias_count == alias_count_new:
                raise ValueError(f'ParameterAlias cycle detected\n{params}')
            alias_count = alias_count_new
        return params