import json
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
Generate text from the eas service.