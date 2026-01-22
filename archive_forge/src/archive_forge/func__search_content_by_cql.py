import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def _search_content_by_cql(self, cql: str, include_archived_spaces: Optional[bool]=None, **kwargs: Any) -> List[dict]:
    url = 'rest/api/content/search'
    params: Dict[str, Any] = {'cql': cql}
    params.update(kwargs)
    if include_archived_spaces is not None:
        params['includeArchivedSpaces'] = include_archived_spaces
    response = self.confluence.get(url, params=params)
    return response.get('results', [])