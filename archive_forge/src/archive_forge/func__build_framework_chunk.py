import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_community.document_loaders.base import BaseLoader
def _build_framework_chunk(dg_chunk: Chunk) -> Document:
    _hashed_id = hashlib.md5(dg_chunk.text.encode()).hexdigest()
    metadata = {XPATH_KEY: dg_chunk.xpath, ID_KEY: _hashed_id, DOCUMENT_NAME_KEY: document_name, DOCUMENT_SOURCE_KEY: document_name, STRUCTURE_KEY: dg_chunk.structure, TAG_KEY: dg_chunk.tag}
    text = dg_chunk.text
    if additional_doc_metadata:
        if self.include_project_metadata_in_doc_metadata:
            metadata.update(additional_doc_metadata)
    return Document(page_content=text[:self.max_text_length], metadata=metadata)