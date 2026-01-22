from typing import Any, cast, Dict, Optional, Sequence, Union
from pyquil import Program
from pyquil.api import QuantumComputer, QuantumExecutable
from pyquil.quilbase import Declare
import cirq
import sympy
from typing_extensions import Protocol
from cirq_rigetti.logging import logger
from cirq_rigetti import circuit_transformers as transformers
def _prepend_real_declarations(*, program: Program, resolvers: Sequence[cirq.ParamResolverOrSimilarType]) -> Program:
    """Adds memory declarations for all variables in each of the `resolver`'s
    param dict. Note, this function assumes that the first parameter resolver
    will contain all variables subsequently referenced in `resolvers`.

    Args:
        program: The program that the quantum computer will execute.
        resolvers: A sequence of parameters resolvers that provide values
            for parameter references in the original `cirq.Circuit`.
    Returns:
        A program that includes the QUIL memory declarations as described above.
    """
    if len(resolvers) > 0:
        resolver = resolvers[0]
        param_dict = _get_param_dict(resolver)
        for key in param_dict.keys():
            declaration = Declare(str(key), 'REAL')
            program._instructions.insert(0, declaration)
            program._synthesized_instructions = None
            program.declarations[declaration.name] = declaration
            logger.debug(f'prepended declaration {declaration}')
    return program