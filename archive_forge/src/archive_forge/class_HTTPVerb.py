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
class HTTPVerb(str, Enum):
    """Enumerator of the HTTP verbs."""
    GET = 'get'
    PUT = 'put'
    POST = 'post'
    DELETE = 'delete'
    OPTIONS = 'options'
    HEAD = 'head'
    PATCH = 'patch'
    TRACE = 'trace'

    @classmethod
    def from_str(cls, verb: str) -> HTTPVerb:
        """Parse an HTTP verb."""
        try:
            return cls(verb)
        except ValueError:
            raise ValueError(f'Invalid HTTP verb. Valid values are {cls.__members__}')