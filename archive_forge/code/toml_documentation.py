import json
from pathlib import Path
from typing import Iterator, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Lazily load the TOML documents from the source file or directory.