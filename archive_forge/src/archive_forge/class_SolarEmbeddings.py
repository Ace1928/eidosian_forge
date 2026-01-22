from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional
import requests
from langchain_core._api import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
@deprecated(since='0.0.34', removal='0.2.0', alternative_import='langchain_upstage.ChatUpstage')
class SolarEmbeddings(BaseModel, Embeddings):
    """Solar's embedding service.

    To use, you should have the environment variable``SOLAR_API_KEY`` set
    with your API token, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SolarEmbeddings
            embeddings = SolarEmbeddings()

            query_text = "This is a test query."
            query_result = embeddings.embed_query(query_text)

            document_text = "This is a test document."
            document_result = embeddings.embed_documents([document_text])

    """
    endpoint_url: str = 'https://api.upstage.ai/v1/solar/embeddings'
    'Endpoint URL to use.'
    model: str = 'solar-1-mini-embedding-query'
    'Embeddings model name to use.'
    solar_api_key: Optional[SecretStr] = None
    'API Key for Solar API.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key exists in environment."""
        solar_api_key = convert_to_secret_str(get_from_dict_or_env(values, 'solar_api_key', 'SOLAR_API_KEY'))
        values['solar_api_key'] = solar_api_key
        return values

    def embed(self, text: str) -> List[List[float]]:
        payload = {'model': self.model, 'input': text}
        headers = {'Authorization': f'Bearer {self.solar_api_key.get_secret_value()}', 'Content-Type': 'application/json'}
        response = requests.post(self.endpoint_url, headers=headers, json=payload)
        parsed_response = response.json()
        if len(parsed_response['data']) == 0:
            raise ValueError(f'Solar API returned an error: {parsed_response['base_resp']}')
        embedding = parsed_response['data'][0]['embedding']
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a Solar embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [embed_with_retry(self, text=text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Solar embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = embed_with_retry(self, text=text)
        return embedding