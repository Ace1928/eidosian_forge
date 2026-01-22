from __future__ import annotations
import copy
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import requests
import yaml
from langchain_core.pydantic_v1 import ValidationError
def _get_referenced_request_body(self, ref: Reference) -> Optional[Union[Reference, RequestBody]]:
    """Get a request body (or nested reference) or err."""
    ref_name = ref.ref.split('/')[-1]
    request_bodies = self._request_bodies_strict
    if ref_name not in request_bodies:
        raise ValueError(f'No request body found for {ref_name}')
    return request_bodies[ref_name]