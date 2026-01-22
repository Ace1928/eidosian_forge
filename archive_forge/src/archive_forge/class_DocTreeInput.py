import sys
import os
import re
import codecs
from docutils import TransformSpec
from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
class DocTreeInput(Input):
    """
    Adapter for document tree input.

    The document tree must be passed in the ``source`` parameter.
    """
    default_source_path = 'doctree input'

    def read(self):
        """Return the document tree."""
        return self.source