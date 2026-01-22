import os
from pathlib import Path
from typing import Any, Iterator, List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import (
Initialize with a file path.

        Args:
            file_path: The path to the Outlook Message file.
        