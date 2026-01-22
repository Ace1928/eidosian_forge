from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
class MermaidDrawMethod(Enum):
    """Enum for different draw methods supported by Mermaid"""
    PYPPETEER = 'pyppeteer'
    API = 'api'