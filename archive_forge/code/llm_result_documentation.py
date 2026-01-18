from __future__ import annotations
from copy import deepcopy
from typing import List, Optional
from langchain_core.outputs.generation import Generation
from langchain_core.outputs.run_info import RunInfo
from langchain_core.pydantic_v1 import BaseModel
Check for LLMResult equality by ignoring any metadata related to runs.