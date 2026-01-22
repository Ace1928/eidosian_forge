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
class DefaultSubstitutions(SphinxTransform):
    """
    Replace some substitutions if they aren't defined in the document.
    """
    default_priority = 210

    def apply(self, **kwargs: Any) -> None:
        to_handle = default_substitutions - set(self.document.substitution_defs)
        for ref in self.document.findall(nodes.substitution_reference):
            refname = ref['refname']
            if refname in to_handle:
                text = self.config[refname]
                if refname == 'today' and (not text):
                    text = format_date(self.config.today_fmt or _('%b %d, %Y'), language=self.config.language)
                ref.replace_self(nodes.Text(text))