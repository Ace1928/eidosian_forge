import warnings
from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa
def _build_params_header(self, params):
    if params is None:
        return ''
    if not isinstance(params, dict):
        raise TypeError("'params' must be a dict")
    params_header = 'CYPHER '
    for key, value in params.items():
        params_header += str(key) + '=' + stringify_param_value(value) + ' '
    return params_header