from typing import Any, Dict, List, Optional
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
class BaichuanTextEmbeddings(BaseModel, Embeddings):
    """Baichuan Text Embedding models."""
    session: Any
    model_name: str = 'Baichuan-Text-Embedding'
    baichuan_api_key: Optional[SecretStr] = None

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        try:
            baichuan_api_key = convert_to_secret_str(get_from_dict_or_env(values, 'baichuan_api_key', 'BAICHUAN_API_KEY'))
        except ValueError as original_exc:
            try:
                baichuan_api_key = convert_to_secret_str(get_from_dict_or_env(values, 'baichuan_auth_token', 'BAICHUAN_AUTH_TOKEN'))
            except ValueError:
                raise original_exc
        session = requests.Session()
        session.headers.update({'Authorization': f'Bearer {baichuan_api_key.get_secret_value()}', 'Accept-Encoding': 'identity', 'Content-type': 'application/json'})
        values['session'] = session
        return values

    def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Internal method to call Baichuan Embedding API and return embeddings.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of list of floats representing the embeddings, or None if an
            error occurs.
        """
        try:
            response = self.session.post(BAICHUAN_API_URL, json={'input': texts, 'model': self.model_name})
            if response.status_code == 200:
                resp = response.json()
                embeddings = resp.get('data', [])
                sorted_embeddings = sorted(embeddings, key=lambda e: e.get('index', 0))
                return [result.get('embedding', []) for result in sorted_embeddings]
            else:
                print(f'Error: Received status code {response.status_code} from \n                    embedding API')
                return None
        except Exception as e:
            print(f'Exception occurred while trying to get embeddings: {str(e)}')
            return None

    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Public method to get embeddings for a list of documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embeddings, one for each text, or None if an error occurs.
        """
        return self._embed(texts)

    def embed_query(self, text: str) -> Optional[List[float]]:
        """Public method to get embedding for a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text, or None if an error occurs.
        """
        result = self._embed([text])
        return result[0] if result is not None else None