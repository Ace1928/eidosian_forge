from typing import Any, List, Optional
from langchain_community.document_loaders.sitemap import SitemapLoader
def _parsing_function(self, content: Any) -> str:
    """Parses specific elements from a Docusaurus page."""
    relevant_elements = content.select(','.join(self.custom_html_tags))
    for element in relevant_elements:
        if element not in relevant_elements:
            element.decompose()
    return str(content.get_text())