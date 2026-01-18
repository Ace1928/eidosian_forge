from __future__ import annotations
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.load.serializable import Serializable
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import root_validator
Count approximate number of tokens