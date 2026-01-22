import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import (
from langchain_core.pydantic_v1 import root_validator
@staticmethod
def _format_context(docs: Sequence[Document]) -> List[str]:
    """
        Format the output of the retriever by including
        special ref tags for tracking the metadata after compression
        """
    formatted_docs = []
    for i, doc in enumerate(docs):
        content = doc.page_content.replace('\n\n', '\n')
        doc_string = f'\n\n<#ref{i}#> {content} <#ref{i}#>\n\n'
        formatted_docs.append(doc_string)
    return formatted_docs