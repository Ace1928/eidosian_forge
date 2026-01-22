import functools
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Union
import yaml
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _replace_template_var(self, placeholders: Dict[str, str], match: re.Match) -> str:
    """Replace a template variable with a placeholder."""
    placeholder = f'__TEMPLATE_VAR_{len(placeholders)}__'
    placeholders[placeholder] = match.group(1)
    return placeholder