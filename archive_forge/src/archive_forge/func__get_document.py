from typing import Any, Iterator, List, Optional
from urllib.parse import urljoin, urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders.web_base import WebBaseLoader
def _get_document(self, soup: Any, custom_url: Optional[str]=None) -> Optional[Document]:
    """Fetch content from page and return Document."""
    page_content_raw = soup.find(self.content_selector)
    if not page_content_raw:
        return None
    content = page_content_raw.get_text(separator='\n').strip()
    title_if_exists = page_content_raw.find('h1')
    title = title_if_exists.text if title_if_exists else ''
    metadata = {'source': custom_url or self.web_path, 'title': title}
    return Document(page_content=content, metadata=metadata)