import logging
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator

        Run Wikipedia search and get the article text plus the meta information.
        See

        Returns: a list of documents.

        