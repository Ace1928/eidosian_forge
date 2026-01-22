from qcs_api_client.models import InstructionSetArchitecture, Characteristic, Operation
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
import numpy as np
from pyquil.external.rpcq import (
from typing import List, Union, cast, DefaultDict, Set, Optional
from collections import defaultdict
def _transform_edge_operation_to_gates(operation_name: str, characteristics: List[Characteristic]) -> List[GateInfo]:
    if operation_name == Supported2QGate.CZ:
        return _make_cz_gates(characteristics)
    elif operation_name == Supported2QGate.ISWAP:
        return _make_iswap_gates(characteristics)
    elif operation_name == Supported2QGate.CPHASE:
        return _make_cphase_gates(characteristics)
    elif operation_name == Supported2QGate.XY:
        return _make_xy_gates(characteristics)
    elif operation_name == Supported2QGate.WILDCARD:
        return _make_wildcard_2q_gates()
    else:
        raise QCSISAParseError('Unsupported edge operation: {}'.format(operation_name))