from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_clean_text(element: Tag) -> str:
    """Returns cleaned text with newlines preserved and irrelevant elements removed."""
    elements_to_skip = ['script', 'noscript', 'canvas', 'meta', 'svg', 'map', 'area', 'audio', 'source', 'track', 'video', 'embed', 'object', 'param', 'picture', 'iframe', 'frame', 'frameset', 'noframes', 'applet', 'form', 'button', 'select', 'base', 'style', 'img']
    newline_elements = ['p', 'div', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'table', 'tr']
    text = _process_element(element, elements_to_skip, newline_elements)
    return text.strip()