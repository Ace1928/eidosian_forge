from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def text_to_docs(text: Union[str, List[str]]) -> List[Document]:
    """Convert a string or list of strings to a list of Documents with metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, separators=['\n\n', '\n', '.', '!', '?', ',', ' ', ''], chunk_overlap=20)
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata['page'] = i + 1
    doc_chunks = []
    for doc in page_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={'page': doc.metadata['page'], 'chunk': i})
            doc.metadata['source'] = f'{doc.metadata['page']}-{doc.metadata['chunk']}'
            doc_chunks.append(doc)
    return doc_chunks