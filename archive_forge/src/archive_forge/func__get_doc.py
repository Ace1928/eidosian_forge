import os
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.base import BaseLoader
def _get_doc(self, cli: Any, field_content: Any, d_id: Union[str, int]) -> Document:
    content = ''
    doc = cli.get_document(d_id)
    content += f'<h2>{doc['name']}<h2/>'
    for f in doc['fields']:
        content += f'{f['name']}\n'
        fc = field_content(f['content'])
        content += fc.get_text()
        content += '\n'
    return Document(metadata={'source': f'rspace: {doc['name']}-{doc['globalId']}'}, page_content=content)