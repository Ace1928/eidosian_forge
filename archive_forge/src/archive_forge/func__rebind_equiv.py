import copy
from collections import namedtuple
from rustworkx.visualization import graphviz_draw
import rustworkx as rx
from qiskit.exceptions import InvalidFileError
from .exceptions import CircuitError
from .parameter import Parameter
from .parameterexpression import ParameterExpression
def _rebind_equiv(equiv, query_params):
    equiv_params, equiv_circuit = equiv
    param_map = {x: y for x, y in zip(equiv_params, query_params) if isinstance(x, Parameter)}
    equiv = equiv_circuit.assign_parameters(param_map, inplace=False, flat_input=True)
    return equiv