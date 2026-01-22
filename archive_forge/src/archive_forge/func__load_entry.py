import logging
import re
from pathlib import Path
from typing import Any, Iterator, List, Mapping, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.bibtex import BibtexparserWrapper
def _load_entry(self, entry: Mapping[str, Any]) -> Optional[Document]:
    import fitz
    parent_dir = Path(self.file_path).parent
    file_names = self.file_regex.findall(entry.get('file', ''))
    if not file_names:
        return None
    texts: List[str] = []
    for file_name in file_names:
        try:
            with fitz.open(parent_dir / file_name) as f:
                texts.extend((page.get_text() for page in f))
        except FileNotFoundError as e:
            logger.debug(e)
    content = '\n'.join(texts) or entry.get('abstract', '')
    if self.max_content_chars:
        content = content[:self.max_content_chars]
    metadata = self.parser.get_metadata(entry, load_extra=self.load_extra_metadata)
    return Document(page_content=content, metadata=metadata)