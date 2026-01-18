from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def transform_serialized(serialized: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms the serialized field of a run dictionary to be compatible
                with WBTraceTree.
            :param serialized: The serialized field of a run dictionary.
            :return: The transformed serialized field.
            """
    serialized = handle_id_and_kwargs(serialized, root=True)
    serialized = remove_exact_and_partial_keys(serialized)
    return serialized