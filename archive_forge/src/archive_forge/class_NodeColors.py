from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
@dataclass
class NodeColors:
    """Schema for Hexadecimal color codes for different node types"""
    start: str = '#ffdfba'
    end: str = '#baffc9'
    other: str = '#fad7de'