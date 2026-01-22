import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote
from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def _load_documents(self) -> List[Document]:
    """Load all documents in the folder.

        Returns:
            List of documents.
        """
    loader = DirectoryLoader(path=self.folder_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents