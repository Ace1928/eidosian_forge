from typing import Any, Dict, Iterable, cast
from docutils import nodes
from docutils.nodes import Element, TextElement
from docutils.writers.manpage import Translator as BaseTranslator
from docutils.writers.manpage import Writer
from sphinx import addnodes
from sphinx.builders import Builder
from sphinx.locale import _, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.i18n import format_date
from sphinx.util.nodes import NodeMatcher
class ManualPageWriter(Writer):

    def __init__(self, builder: Builder) -> None:
        super().__init__()
        self.builder = builder

    def translate(self) -> None:
        transform = NestedInlineTransform(self.document)
        transform.apply()
        visitor = self.builder.create_translator(self.document, self.builder)
        self.visitor = cast(ManualPageTranslator, visitor)
        self.document.walkabout(visitor)
        self.output = self.visitor.astext()