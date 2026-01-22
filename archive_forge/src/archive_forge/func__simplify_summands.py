import warnings
import itertools
from copy import copy
from typing import List
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, convert_to_opmath
from pennylane.ops.qubit import Hamiltonian
from pennylane.queuing import QueuingManager
from .composite import CompositeOp
@classmethod
def _simplify_summands(cls, summands: List[Operator]):
    """Reduces the depth of nested summands and groups equal terms together.

        Args:
            summands (List[~.operation.Operator]): summands list to simplify

        Returns:
            .SumSummandsGrouping: Class containing the simplified and grouped summands.
        """
    new_summands = _SumSummandsGrouping()
    for summand in summands:
        if isinstance(summand, Sum):
            sum_summands = cls._simplify_summands(summands=summand.operands)
            for op_hash, [coeff, sum_summand] in sum_summands.queue.items():
                new_summands.add(summand=sum_summand, coeff=coeff, op_hash=op_hash)
            continue
        simplified_summand = summand.simplify()
        if isinstance(simplified_summand, Sum):
            sum_summands = cls._simplify_summands(summands=simplified_summand.operands)
            for op_hash, [coeff, sum_summand] in sum_summands.queue.items():
                new_summands.add(summand=sum_summand, coeff=coeff, op_hash=op_hash)
        else:
            new_summands.add(summand=simplified_summand)
    return new_summands