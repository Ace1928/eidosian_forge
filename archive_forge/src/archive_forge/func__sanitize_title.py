import logging
import re
import xml.etree.cElementTree
import xml.sax.saxutils
from io import BytesIO
from typing import List, Optional, Sequence
from xml.etree.ElementTree import ElementTree
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def _sanitize_title(title: str) -> str:
    sanitized_title = re.sub('\\s', ' ', title)
    sanitized_title = re.sub('(?u)[^- \\w.]', '', sanitized_title)
    if len(sanitized_title) > _MAXIMUM_TITLE_LENGTH:
        sanitized_title = sanitized_title[:_MAXIMUM_TITLE_LENGTH]
    return sanitized_title