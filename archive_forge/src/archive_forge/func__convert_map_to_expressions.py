from typing import Dict, Union, Optional, TYPE_CHECKING
from ray.util.annotations import PublicAPI
def _convert_map_to_expressions(map_expressions: LabelMatchExpressionsT, param: str):
    expressions = []
    if map_expressions is None:
        return expressions
    if not isinstance(map_expressions, Dict):
        raise ValueError(f'The {param} parameter must be a map (e.g. {{"key1": In("value1")}}) but got type {type(map_expressions)}.')
    for key, value in map_expressions.items():
        if not isinstance(key, str):
            raise ValueError(f'The map key of the {param} parameter must be of type str (e.g. {{"key1": In("value1")}}) but got {str(key)} of type {type(key)}.')
        if not isinstance(value, (In, NotIn, Exists, DoesNotExist)):
            raise ValueError(f'The map value for key {key} of the {param} parameter must be one of the `In`, `NotIn`, `Exists` or `DoesNotExist` operator (e.g. {{"key1": In("value1")}}) but got {str(value)} of type {type(value)}.')
        expressions.append(_LabelMatchExpression(key, value))
    return expressions