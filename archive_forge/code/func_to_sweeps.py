from typing import Iterable, Iterator, List, Sequence, Union, cast
import warnings
from typing_extensions import Protocol
from cirq._doc import document
from cirq.study.resolver import ParamResolver, ParamResolverOrSimilarType
from cirq.study.sweeps import ListSweep, Points, Sweep, UnitSweep, Zip, dict_to_product_sweep
def to_sweeps(sweepable: Sweepable) -> List[Sweep]:
    """Converts a Sweepable to a list of Sweeps."""
    if sweepable is None:
        return [UnitSweep]
    if isinstance(sweepable, ParamResolver):
        return [_resolver_to_sweep(sweepable)]
    if isinstance(sweepable, Sweep):
        return [sweepable]
    if isinstance(sweepable, dict):
        if any((isinstance(val, Sequence) for val in sweepable.values())):
            warnings.warn('Implicit expansion of a dictionary into a Cartesian product of sweeps is deprecated and will be removed in cirq 0.10. Instead, expand the sweep explicitly using `cirq.dict_to_product_sweep`.', DeprecationWarning, stacklevel=2)
        product_sweep = dict_to_product_sweep(sweepable)
        return [_resolver_to_sweep(resolver) for resolver in product_sweep]
    if isinstance(sweepable, Iterable) and (not isinstance(sweepable, str)):
        return [sweep for item in sweepable for sweep in to_sweeps(item)]
    raise TypeError(f'Unrecognized sweepable type: {type(sweepable)}.\nsweepable: {sweepable}')