from __future__ import annotations
import json
import logging
import warnings
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

        Function that does a dummy embedding to figure out how many dimensions
        this embedding function returns. Needed for the virtual table DDL.
        