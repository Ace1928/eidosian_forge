import os
import posixpath
import re
import urllib.parse
import warnings
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node, Text
from docutils.writers.html4css1 import HTMLTranslator as BaseTranslator
from docutils.writers.html4css1 import Writer
from sphinx import addnodes
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.images import get_image_size
def visit_productionlist(self, node: Element) -> None:
    self.body.append(self.starttag(node, 'pre'))
    names = []
    productionlist = cast(Iterable[addnodes.production], node)
    for production in productionlist:
        names.append(production['tokenname'])
    maxlen = max((len(name) for name in names))
    lastname = None
    for production in productionlist:
        if production['tokenname']:
            lastname = production['tokenname'].ljust(maxlen)
            self.body.append(self.starttag(production, 'strong', ''))
            self.body.append(lastname + '</strong> ::= ')
        elif lastname is not None:
            self.body.append('%s     ' % (' ' * len(lastname)))
        production.walkabout(self)
        self.body.append('\n')
    self.body.append('</pre>\n')
    raise nodes.SkipNode