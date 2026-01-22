import os
import re
import sys
from typing import Any, Dict, List
from sphinx.errors import ExtensionError, SphinxError
from sphinx.search import SearchLanguage
from sphinx.util import import_object
class BaseSplitter:

    def __init__(self, options: Dict) -> None:
        self.options = options

    def split(self, input: str) -> List[str]:
        """
        :param str input:
        :return:
        :rtype: list[str]
        """
        raise NotImplementedError