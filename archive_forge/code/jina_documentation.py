from typing import Any, Dict, List, Optional
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
Call out to Jina's embedding endpoint.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        