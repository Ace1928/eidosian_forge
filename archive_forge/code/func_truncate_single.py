from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def truncate_single(run: Dict[str, Any]) -> Dict[str, Any]:
    """Utility to truncate a single run dictionary to only keep the specified
                keys.
            :param run: The run dictionary to truncate.
            :return: The truncated run dictionary
            """
    new_dict = {}
    for key in run:
        if key in keep_keys:
            new_dict[key] = run.get(key)
    return new_dict