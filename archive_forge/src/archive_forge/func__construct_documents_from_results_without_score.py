from __future__ import annotations
import json
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _construct_documents_from_results_without_score(self, results: Dict[str, List[Dict[str, str]]]) -> List[Document]:
    """Helper to convert Marqo results into documents.

        Args:
            results (List[dict]): A marqo results object with the 'hits'.
            include_scores (bool, optional): Include scores alongside documents.
            Defaults to False.

        Returns:
            Union[List[Document], List[Tuple[Document, float]]]: The documents or
            document score pairs if `include_scores` is true.
        """
    documents: List[Document] = []
    for res in results['hits']:
        if self.page_content_builder is None:
            text = res['text']
        else:
            text = self.page_content_builder(res)
        metadata = json.loads(res.get('metadata', '{}'))
        documents.append(Document(page_content=text, metadata=metadata))
    return documents