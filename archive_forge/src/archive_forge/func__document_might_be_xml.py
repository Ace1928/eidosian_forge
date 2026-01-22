from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
def _document_might_be_xml(self, processing_instruction):
    """Call this method when encountering an XML declaration, or a
        "processing instruction" that might be an XML declaration.
        """
    if self._first_processing_instruction is not None or self._root_tag is not None:
        return
    self._first_processing_instruction = processing_instruction