import hashlib
import logging
from base64 import b64decode
from pathlib import Path
from time import strptime
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def _parse_note_xml(xml_file: str) -> Iterator[Dict[str, Any]]:
    """Parse Evernote xml."""
    try:
        from lxml import etree
    except ImportError as e:
        logger.error('Could not import `lxml`. Although it is not a required package to use Langchain, using the EverNote loader requires `lxml`. Please install `lxml` via `pip install lxml` and try again.')
        raise e
    context = etree.iterparse(xml_file, encoding='utf-8', strip_cdata=False, huge_tree=True, recover=True)
    for action, elem in context:
        if elem.tag == 'note':
            yield EverNoteLoader._parse_note(elem)