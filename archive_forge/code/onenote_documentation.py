from pathlib import Path
from typing import Dict, Iterator, List, Optional
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_community.document_loaders.base import BaseLoader
Create URL for getting page ids from the OneNoteApi API.