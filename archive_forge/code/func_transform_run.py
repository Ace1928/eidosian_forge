from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def transform_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms a run dictionary to be compatible with WBTraceTree.
            :param run: The run dictionary to transform.
            :return: The transformed run dictionary.
            """
    transformed_dict = transform_serialized(run)
    serialized = transformed_dict.pop('serialized')
    for k, v in serialized.items():
        transformed_dict[k] = v
    _kind = transformed_dict.get('_kind', None)
    name = transformed_dict.pop('name', None)
    exec_ord = transformed_dict.pop('execution_order', None)
    if not name:
        name = _kind
    output_dict = {f'{exec_ord}_{name}': transformed_dict}
    return output_dict