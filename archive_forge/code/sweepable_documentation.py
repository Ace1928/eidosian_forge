from typing import Iterable, Iterator, List, Sequence, Union, cast
import warnings
from typing_extensions import Protocol
from cirq._doc import document
from cirq.study.resolver import ParamResolver, ParamResolverOrSimilarType
from cirq.study.sweeps import ListSweep, Points, Sweep, UnitSweep, Zip, dict_to_product_sweep
Converts the argument into a ``cirq.Sweep``.

    Args:
        sweep_or_resolver_list: The object to try to turn into a
            ``cirq.Sweep`` . A ``cirq.Sweep``, a single ``cirq.ParamResolver``,
            or a list of ``cirq.ParamResolver`` s.

    Returns:
        A sweep equal to or containing the argument.

    Raises:
        TypeError: If an unsupport type was supplied.
    