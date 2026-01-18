import warnings
from typing import Any, Dict, List
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.vectorstores import VectorStore
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
Extra search args.