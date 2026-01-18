from typing import Iterable, Iterator, List, Sequence, Union, cast
import warnings
from typing_extensions import Protocol
from cirq._doc import document
from cirq.study.resolver import ParamResolver, ParamResolverOrSimilarType
from cirq.study.sweeps import ListSweep, Points, Sweep, UnitSweep, Zip, dict_to_product_sweep
def to_resolvers(sweepable: Sweepable) -> Iterator[ParamResolver]:
    """Convert a Sweepable to a list of ParamResolvers."""
    for sweep in to_sweeps(sweepable):
        yield from sweep