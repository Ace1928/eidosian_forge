import functools
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Union
import yaml
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _parse_document_tags(self, content: str) -> set:
    """Return a set of all tags in within the document."""
    if not self.collect_metadata:
        return set()
    match = self.TAG_REGEX.findall(content)
    if not match:
        return set()
    return {tag for tag in match}