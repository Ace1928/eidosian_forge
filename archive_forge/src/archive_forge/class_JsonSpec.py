from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.callbacks import (
from langchain_core.tools import BaseTool
class JsonSpec(BaseModel):
    """Base class for JSON spec."""
    dict_: Dict
    max_value_length: int = 200

    @classmethod
    def from_file(cls, path: Path) -> JsonSpec:
        """Create a JsonSpec from a file."""
        if not path.exists():
            raise FileNotFoundError(f'File not found: {path}')
        dict_ = json.loads(path.read_text())
        return cls(dict_=dict_)

    def keys(self, text: str) -> str:
        """Return the keys of the dict at the given path.

        Args:
            text: Python representation of the path to the dict (e.g. data["key1"][0]["key2"]).
        """
        try:
            items = _parse_input(text)
            val = self.dict_
            for i in items:
                if i:
                    val = val[i]
            if not isinstance(val, dict):
                raise ValueError(f'Value at path `{text}` is not a dict, get the value directly.')
            return str(list(val.keys()))
        except Exception as e:
            return repr(e)

    def value(self, text: str) -> str:
        """Return the value of the dict at the given path.

        Args:
            text: Python representation of the path to the dict (e.g. data["key1"][0]["key2"]).
        """
        try:
            items = _parse_input(text)
            val = self.dict_
            for i in items:
                val = val[i]
            if isinstance(val, dict) and len(str(val)) > self.max_value_length:
                return 'Value is a large dictionary, should explore its keys directly'
            str_val = str(val)
            if len(str_val) > self.max_value_length:
                str_val = str_val[:self.max_value_length] + '...'
            return str_val
        except Exception as e:
            return repr(e)