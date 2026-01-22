from pathlib import Path
from typing import Dict, Iterator, List, Optional
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_community.document_loaders.base import BaseLoader
class _OneNoteGraphSettings(BaseSettings):
    client_id: str = Field(..., env='MS_GRAPH_CLIENT_ID')
    client_secret: SecretStr = Field(..., env='MS_GRAPH_CLIENT_SECRET')

    class Config:
        """Config for OneNoteGraphSettings."""
        env_prefix = ''
        case_sentive = False
        env_file = '.env'