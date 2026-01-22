import os
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.base import BaseLoader
def _load_folder_tree(self) -> Iterator[Document]:
    cli, field_content = self._create_rspace_client()
    if self.global_id:
        docs_in_folder = cli.list_folder_tree(folder_id=self.global_id[2:], typesToInclude=['document'])
    doc_ids: List[int] = [d['id'] for d in docs_in_folder['records']]
    for doc_id in doc_ids:
        yield self._get_doc(cli, field_content, doc_id)