from typing import Iterable, Iterator, List, Sequence, Union, cast
import warnings
from typing_extensions import Protocol
from cirq._doc import document
from cirq.study.resolver import ParamResolver, ParamResolverOrSimilarType
from cirq.study.sweeps import ListSweep, Points, Sweep, UnitSweep, Zip, dict_to_product_sweep
def _resolver_to_sweep(resolver: ParamResolver) -> Sweep:
    params = resolver.param_dict
    if not params:
        return UnitSweep
    return Zip(*[Points(key, [cast(float, value)]) for key, value in params.items()])