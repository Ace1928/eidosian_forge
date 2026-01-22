import logging
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def _page_to_document(self, page_title: str, wiki_page: Any) -> Document:
    main_meta = {'title': page_title, 'summary': wiki_page.summary, 'source': wiki_page.url}
    add_meta = {'categories': wiki_page.categories, 'page_url': wiki_page.url, 'image_urls': wiki_page.images, 'related_titles': wiki_page.links, 'parent_id': wiki_page.parent_id, 'references': wiki_page.references, 'revision_id': wiki_page.revision_id, 'sections': wiki_page.sections} if self.load_all_available_meta else {}
    doc = Document(page_content=wiki_page.content[:self.doc_content_chars_max], metadata={**main_meta, **add_meta})
    return doc