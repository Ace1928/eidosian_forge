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
class TagFieldSchema(RedisField):
    """Schema for tag fields in Redis."""
    separator: str = ','
    case_sensitive: bool = False
    no_index: bool = False
    sortable: Optional[bool] = False

    def as_field(self) -> TagField:
        from redis.commands.search.field import TagField
        return TagField(self.name, separator=self.separator, case_sensitive=self.case_sensitive, sortable=self.sortable, no_index=self.no_index)