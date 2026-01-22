from __future__ import annotations
import json
from typing import Dict, List, Optional
import aiohttp
import requests
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env, get_from_env
class AzureCognitiveSearchRetriever(AzureAISearchRetriever):
    """`Azure Cognitive Search` service retriever.
    This version of the retriever will soon be
    depreciated. Please switch to AzureAISearchRetriever
    """