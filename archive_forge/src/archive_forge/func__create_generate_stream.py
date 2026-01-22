import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Union
import aiohttp
import requests
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra
def _create_generate_stream(self, prompt: str, stop: Optional[List[str]]=None, images: Optional[List[str]]=None, **kwargs: Any) -> Iterator[str]:
    payload = {'prompt': prompt, 'images': images}
    yield from self._create_stream(payload=payload, stop=stop, api_url=f'{self.base_url}/api/generate', **kwargs)