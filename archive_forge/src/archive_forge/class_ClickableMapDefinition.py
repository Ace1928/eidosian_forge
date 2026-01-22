import posixpath
import re
import subprocess
from os import path
from subprocess import PIPE, CalledProcessError
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging, sha1
from sphinx.util.docutils import SphinxDirective, SphinxTranslator
from sphinx.util.fileutil import copy_asset
from sphinx.util.i18n import search_image_for_language
from sphinx.util.nodes import set_source_info
from sphinx.util.osutil import ensuredir
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.manpage import ManualPageTranslator
from sphinx.writers.texinfo import TexinfoTranslator
from sphinx.writers.text import TextTranslator
class ClickableMapDefinition:
    """A manipulator for clickable map file of graphviz."""
    maptag_re = re.compile('<map id="(.*?)"')
    href_re = re.compile('href=".*?"')

    def __init__(self, filename: str, content: str, dot: str='') -> None:
        self.id: Optional[str] = None
        self.filename = filename
        self.content = content.splitlines()
        self.clickable: List[str] = []
        self.parse(dot=dot)

    def parse(self, dot: str) -> None:
        matched = self.maptag_re.match(self.content[0])
        if not matched:
            raise GraphvizError('Invalid clickable map file found: %s' % self.filename)
        self.id = matched.group(1)
        if self.id == '%3':
            hashed = sha1(dot.encode()).hexdigest()
            self.id = 'grapviz%s' % hashed[-10:]
            self.content[0] = self.content[0].replace('%3', self.id)
        for line in self.content:
            if self.href_re.search(line):
                self.clickable.append(line)

    def generate_clickable_map(self) -> str:
        """Generate clickable map tags if clickable item exists.

        If not exists, this only returns empty string.
        """
        if self.clickable:
            return '\n'.join([self.content[0]] + self.clickable + [self.content[-1]])
        else:
            return ''