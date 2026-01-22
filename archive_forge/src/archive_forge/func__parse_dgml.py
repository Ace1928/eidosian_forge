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
def _parse_dgml(self, content: bytes, document_name: Optional[str]=None, additional_doc_metadata: Optional[Mapping]=None) -> List[Document]:
    """Parse a single DGML document into a list of Documents."""
    try:
        from lxml import etree
    except ImportError:
        raise ImportError('Could not import lxml python package. Please install it with `pip install lxml`.')
    try:
        from dgml_utils.models import Chunk
        from dgml_utils.segmentation import get_chunks
    except ImportError:
        raise ImportError('Could not import from dgml-utils python package. Please install it with `pip install dgml-utils`.')

    def _build_framework_chunk(dg_chunk: Chunk) -> Document:
        _hashed_id = hashlib.md5(dg_chunk.text.encode()).hexdigest()
        metadata = {XPATH_KEY: dg_chunk.xpath, ID_KEY: _hashed_id, DOCUMENT_NAME_KEY: document_name, DOCUMENT_SOURCE_KEY: document_name, STRUCTURE_KEY: dg_chunk.structure, TAG_KEY: dg_chunk.tag}
        text = dg_chunk.text
        if additional_doc_metadata:
            if self.include_project_metadata_in_doc_metadata:
                metadata.update(additional_doc_metadata)
        return Document(page_content=text[:self.max_text_length], metadata=metadata)
    tree = etree.parse(io.BytesIO(content))
    root = tree.getroot()
    dg_chunks = get_chunks(root, min_text_length=self.min_text_length, max_text_length=self.max_text_length, whitespace_normalize_text=self.whitespace_normalize_text, sub_chunk_tables=self.sub_chunk_tables, include_xml_tags=self.include_xml_tags, parent_hierarchy_levels=self.parent_hierarchy_levels)
    framework_chunks: Dict[str, Document] = {}
    for dg_chunk in dg_chunks:
        framework_chunk = _build_framework_chunk(dg_chunk)
        chunk_id = framework_chunk.metadata.get(ID_KEY)
        if chunk_id:
            framework_chunks[chunk_id] = framework_chunk
            if dg_chunk.parent:
                framework_parent_chunk = _build_framework_chunk(dg_chunk.parent)
                parent_id = framework_parent_chunk.metadata.get(ID_KEY)
                if parent_id and framework_parent_chunk.page_content:
                    framework_chunk.metadata[self.parent_id_key] = parent_id
                    framework_chunks[parent_id] = framework_parent_chunk
    return list(framework_chunks.values())