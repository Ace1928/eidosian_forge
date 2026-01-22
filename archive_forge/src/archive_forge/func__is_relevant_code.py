import re
from typing import Callable, List
from langchain_community.document_loaders.parsers.language.code_segmenter import (
def _is_relevant_code(self, line: str) -> bool:
    """Check if a line is part of the procedure division or a relevant section."""
    if 'PROCEDURE DIVISION' in line.upper():
        return True
    return False