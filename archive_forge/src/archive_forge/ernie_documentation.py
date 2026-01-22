import asyncio
import logging
import threading
from typing import Dict, List, Optional
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import get_from_dict_or_env
Asynchronous Embed search docs.

        Args:
            texts: The list of texts to embed

        Returns:
            List[List[float]]: List of embeddings, one for each text.
        