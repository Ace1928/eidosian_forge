import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
Call out to HuggingFace Hub's inference endpoint.