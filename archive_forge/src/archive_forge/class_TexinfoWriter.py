import re
import textwrap
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Pattern, Set,
from docutils import nodes, writers
from docutils.nodes import Element, Node, Text
from sphinx import __display_version__, addnodes
from sphinx.domains import IndexEntry
from sphinx.domains.index import IndexDomain
from sphinx.errors import ExtensionError
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.i18n import format_date
from sphinx.writers.latex import collected_footnote
class TexinfoWriter(writers.Writer):
    """Texinfo writer for generating Texinfo documents."""
    supported = ('texinfo', 'texi')
    settings_spec: Tuple[str, Any, Tuple[Tuple[str, List[str], Dict[str, str]], ...]] = ('Texinfo Specific Options', None, (('Name of the Info file', ['--texinfo-filename'], {'default': ''}), ('Dir entry', ['--texinfo-dir-entry'], {'default': ''}), ('Description', ['--texinfo-dir-description'], {'default': ''}), ('Category', ['--texinfo-dir-category'], {'default': 'Miscellaneous'})))
    settings_defaults: Dict = {}
    output: Optional[str] = None
    visitor_attributes = ('output', 'fragment')

    def __init__(self, builder: 'TexinfoBuilder') -> None:
        super().__init__()
        self.builder = builder

    def translate(self) -> None:
        visitor = self.builder.create_translator(self.document, self.builder)
        self.visitor = cast(TexinfoTranslator, visitor)
        self.document.walkabout(visitor)
        self.visitor.finish()
        for attr in self.visitor_attributes:
            setattr(self, attr, getattr(self.visitor, attr))