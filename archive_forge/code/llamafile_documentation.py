from __future__ import annotations
import json
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional
import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra
from langchain_core.utils import get_pydantic_field_names
When streaming is turned on, llamafile server returns lines like:

        'data: {"content":" They","multimodal":true,"slot_id":0,"stop":false}'

        Here, we convert this to a dict and return the value of the 'content'
        field
        