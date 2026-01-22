import json
import logging
from typing import Any, Dict, Iterator, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
def _call_api(self, prompt: str, params: Dict[str, Any]) -> requests.Response:
    """Call Cloudflare Workers API"""
    headers = {'Authorization': f'Bearer {self.api_token}'}
    data = {'prompt': prompt, 'stream': self.streaming, **params}
    response = requests.post(self.endpoint_url, headers=headers, json=data)
    return response