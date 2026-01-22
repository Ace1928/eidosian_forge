from typing import Any, Dict, List, Mapping, Optional
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from requests.adapters import HTTPAdapter, Retry
from typing_extensions import NotRequired, TypedDict
class EmbaasEmbeddingsPayload(TypedDict):
    """Payload for the Embaas embeddings API."""
    model: str
    texts: List[str]
    instruction: NotRequired[str]