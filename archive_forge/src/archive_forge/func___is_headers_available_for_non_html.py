import logging
from typing import Any, List
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def __is_headers_available_for_non_html(self) -> bool:
    _unstructured_version = self.__version.split('-')[0]
    unstructured_version = tuple([int(x) for x in _unstructured_version.split('.')])
    return unstructured_version >= (0, 5, 13)