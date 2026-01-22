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
@property
def _schemas_strict(self) -> Dict[str, Schema]:
    """Get the dictionary of schemas or err."""
    schemas = self._components_strict.schemas
    if schemas is None:
        raise ValueError('No schemas found in spec. ')
    return schemas