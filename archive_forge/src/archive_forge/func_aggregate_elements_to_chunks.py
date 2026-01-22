from __future__ import annotations
import pathlib
from io import BytesIO, StringIO
from typing import Any, Dict, List, Tuple, TypedDict
import requests
from langchain_core.documents import Document
def aggregate_elements_to_chunks(self, elements: List[ElementType]) -> List[Document]:
    """Combine elements with common metadata into chunks

        Args:
            elements: HTML element content with associated identifying info and metadata
        """
    aggregated_chunks: List[ElementType] = []
    for element in elements:
        if aggregated_chunks and aggregated_chunks[-1]['metadata'] == element['metadata']:
            aggregated_chunks[-1]['content'] += '  \n' + element['content']
        else:
            aggregated_chunks.append(element)
    return [Document(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in aggregated_chunks]