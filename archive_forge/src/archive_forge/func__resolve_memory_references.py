from dataclasses import dataclass
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Union, cast
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import ParameterAref, ParameterSpec
from pyquil.api import QuantumExecutable, EncryptedProgram, EngagementManager
from pyquil._memory import Memory
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qpu_client import GetBuffersRequest, QPUClient, BufferResponse, RunProgramRequest
from pyquil.quilatom import (
@classmethod
def _resolve_memory_references(cls, expression: ExpressionDesignator, memory: Memory) -> Union[float, int]:
    """
        Traverse the given Expression, and replace any Memory References with whatever values
        have been so far provided by the user for those memory spaces. Declared memory defaults
        to zero.

        :param expression: an Expression
        """
    if isinstance(expression, BinaryExp):
        left = cls._resolve_memory_references(expression.op1, memory=memory)
        right = cls._resolve_memory_references(expression.op2, memory=memory)
        return cast(Union[float, int], expression.fn(left, right))
    elif isinstance(expression, Function):
        return cast(Union[float, int], expression.fn(cls._resolve_memory_references(expression.expression, memory=memory)))
    elif isinstance(expression, Parameter):
        raise ValueError(f'Unexpected Parameter in gate expression: {expression}')
    elif isinstance(expression, (float, int)):
        return expression
    elif isinstance(expression, MemoryReference):
        return memory.values.get(ParameterAref(name=expression.name, index=expression.offset), 0)
    else:
        raise ValueError(f'Unexpected expression in gate parameter: {expression}')