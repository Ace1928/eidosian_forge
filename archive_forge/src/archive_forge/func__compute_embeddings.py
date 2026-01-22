from typing import Any, Dict, List, Mapping, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _compute_embeddings(self, texts: List[str], instruction: str) -> List[List[float]]:
    """Compute embeddings using an OctoAI instruct model."""
    from octoai import client
    embedding = []
    embeddings = []
    octoai_client = client.Client(token=self.octoai_api_token)
    for text in texts:
        parameter_payload = {'sentence': str([text]), 'input': str([text]), 'instruction': str([instruction]), 'model': 'thenlper/gte-large', 'parameters': self.model_kwargs or {}}
        try:
            resp_json = octoai_client.infer(self.endpoint_url, parameter_payload)
            if 'embeddings' in resp_json:
                embedding = resp_json['embeddings']
            elif 'data' in resp_json:
                json_data = resp_json['data']
                for item in json_data:
                    if 'embedding' in item:
                        embedding = item['embedding']
        except Exception as e:
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e
        embeddings.append(embedding)
    return embeddings