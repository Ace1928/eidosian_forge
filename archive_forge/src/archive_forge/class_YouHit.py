from typing import Any, Dict, List, Literal, Optional
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
class YouHit(YouHitMetadata):
    """A single hit from you.com, which may contain multiple snippets"""
    snippets: List[str] = Field(description='One or snippets of text')