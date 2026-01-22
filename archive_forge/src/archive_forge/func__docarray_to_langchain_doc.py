from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _docarray_to_langchain_doc(self, doc: Union[Dict[str, Any], Any]) -> Document:
    """
        Convert a DocArray document (which also might be a dict)
        to a langchain document format.

        DocArray document can contain arbitrary fields, so the mapping is done
        in the following way:

        page_content <-> content_field
        metadata <-> all other fields excluding
            tensors and embeddings (so float, int, string)

        Args:
            doc: DocArray document

        Returns:
            Document in langchain format

        Raises:
            ValueError: If the document doesn't contain the content field
        """
    fields = doc.keys() if isinstance(doc, dict) else doc.__fields__
    if self.content_field not in fields:
        raise ValueError(f'Document does not contain the content field - {self.content_field}.')
    lc_doc = Document(page_content=doc[self.content_field] if isinstance(doc, dict) else getattr(doc, self.content_field))
    for name in fields:
        value = doc[name] if isinstance(doc, dict) else getattr(doc, name)
        if isinstance(value, (str, int, float, bool)) and name != self.content_field:
            lc_doc.metadata[name] = value
    return lc_doc