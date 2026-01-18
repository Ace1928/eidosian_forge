import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Check if the metadata_func output is valid