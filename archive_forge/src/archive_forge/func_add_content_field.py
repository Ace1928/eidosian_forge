from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing_extensions import TYPE_CHECKING, Literal
from langchain_community.vectorstores.redis.constants import REDIS_VECTOR_DTYPE_MAP
def add_content_field(self) -> None:
    if self.text is None:
        self.text = []
    for field in self.text:
        if field.name == self.content_key:
            return
    self.text.append(TextFieldSchema(name=self.content_key))