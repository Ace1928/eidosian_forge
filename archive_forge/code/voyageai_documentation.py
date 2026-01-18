from __future__ import annotations
import json
import logging
from typing import (
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
Call out to Voyage Embedding endpoint for embedding general text.

        Args:
            texts: The list of texts to embed.
            input_type: Type of the input text. Default to None, meaning the type is
                unspecified. Other options: query, document.

        Returns:
            Embedding for the text.
        