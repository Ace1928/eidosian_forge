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
class NumericFieldSchema(RedisField):
    """Schema for numeric fields in Redis."""
    no_index: bool = False
    sortable: Optional[bool] = False

    def as_field(self) -> NumericField:
        from redis.commands.search.field import NumericField
        return NumericField(self.name, sortable=self.sortable, no_index=self.no_index)