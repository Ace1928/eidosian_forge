import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def _should_pretty_print(self, indent_level=1):
    """Should this tag be pretty-printed?

        Most of them should, but some (such as <pre> in HTML
        documents) should not.
        """
    return indent_level is not None and (not self.preserve_whitespace_tags or self.name not in self.preserve_whitespace_tags)