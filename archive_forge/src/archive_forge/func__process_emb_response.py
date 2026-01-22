import logging
from typing import Any, Dict, List, Mapping, Optional
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
def _process_emb_response(self, input: str) -> List[float]:
    """Process a response from the API.

        Args:
            response: The response from the API.

        Returns:
            The response as a dictionary.
        """
    headers = {'Content-Type': 'application/json', **(self.headers or {})}
    try:
        res = requests.post(f'{self.base_url}/api/embeddings', headers=headers, json={'model': self.model, 'prompt': input, **self._default_params})
    except requests.exceptions.RequestException as e:
        raise ValueError(f'Error raised by inference endpoint: {e}')
    if res.status_code != 200:
        raise ValueError('Error raised by inference API HTTP code: %s, %s' % (res.status_code, res.text))
    try:
        t = res.json()
        return t['embedding']
    except requests.exceptions.JSONDecodeError as e:
        raise ValueError(f'Error raised by inference API: {e}.\nResponse: {res.text}')