from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
class CurveStyle(Enum):
    """Enum for different curve styles supported by Mermaid"""
    BASIS = 'basis'
    BUMP_X = 'bumpX'
    BUMP_Y = 'bumpY'
    CARDINAL = 'cardinal'
    CATMULL_ROM = 'catmullRom'
    LINEAR = 'linear'
    MONOTONE_X = 'monotoneX'
    MONOTONE_Y = 'monotoneY'
    NATURAL = 'natural'
    STEP = 'step'
    STEP_AFTER = 'stepAfter'
    STEP_BEFORE = 'stepBefore'