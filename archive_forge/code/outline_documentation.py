import logging
from typing import Any, Dict, List, Optional
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env

        Run Outline search and get the document content plus the meta information.

        Returns: a list of documents.

        