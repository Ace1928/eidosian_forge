from __future__ import annotations
from typing import Any, Callable, List, Optional, Protocol, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
def _collapse_docs_func(docs: List[Document], **kwargs: Any) -> str:
    return self._collapse_chain.run(input_documents=docs, callbacks=callbacks, **kwargs)