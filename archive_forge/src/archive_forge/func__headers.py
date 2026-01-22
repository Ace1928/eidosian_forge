import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional
import aiohttp
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.requests import Requests
def _headers(self) -> Dict:
    return {'Authorization': f'bearer {self.deepinfra_api_token}', 'Content-Type': 'application/json'}