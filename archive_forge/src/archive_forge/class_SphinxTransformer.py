import re
import unicodedata
import warnings
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, cast
import docutils
from docutils import nodes
from docutils.nodes import Element, Node, Text
from docutils.transforms import Transform, Transformer
from docutils.transforms.parts import ContentsFilter
from docutils.transforms.universal import SmartQuotes
from docutils.utils import normalize_language_tag
from docutils.utils.smartquotes import smartchars
from sphinx import addnodes
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.docutils import new_document
from sphinx.util.i18n import format_date
from sphinx.util.nodes import NodeMatcher, apply_source_workaround, is_smartquotable
class SphinxTransformer(Transformer):
    """
    A transformer for Sphinx.
    """
    document: nodes.document
    env: Optional['BuildEnvironment'] = None

    def set_environment(self, env: 'BuildEnvironment') -> None:
        self.env = env

    def apply_transforms(self) -> None:
        if isinstance(self.document, nodes.document):
            if not hasattr(self.document.settings, 'env') and self.env:
                self.document.settings.env = self.env
            super().apply_transforms()
        else:
            try:
                document = new_document('')
                if self.env:
                    document.settings.env = self.env
                document += self.document
                self.document = document
                super().apply_transforms()
            finally:
                self.document = self.document[0]