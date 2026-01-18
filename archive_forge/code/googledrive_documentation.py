import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_community.document_loaders.base import BaseLoader
Load documents.