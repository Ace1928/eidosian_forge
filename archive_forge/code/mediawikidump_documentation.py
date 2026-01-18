import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Lazy load from a file path.