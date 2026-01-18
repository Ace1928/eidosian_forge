from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
Lambda value for MMR search.