import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple
import aiohttp
import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
call the embedding of model, async method

        Args:
            model (str): to embedding model
            texts (List[str]): List of sentences to embed.

        Returns:
            List[List[float]]: List of vectors for each sentence
        