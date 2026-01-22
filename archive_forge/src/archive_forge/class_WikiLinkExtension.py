from __future__ import annotations
from . import Extension
from ..inlinepatterns import InlineProcessor
import xml.etree.ElementTree as etree
import re
from typing import Any
class WikiLinkExtension(Extension):
    """ Add inline processor to Markdown. """

    def __init__(self, **kwargs):
        self.config = {'base_url': ['/', 'String to append to beginning or URL.'], 'end_url': ['/', 'String to append to end of URL.'], 'html_class': ['wikilink', 'CSS hook. Leave blank for none.'], 'build_url': [build_url, 'Callable formats URL from label.']}
        ' Default configuration options. '
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        self.md = md
        WIKILINK_RE = '\\[\\[([\\w0-9_ -]+)\\]\\]'
        wikilinkPattern = WikiLinksInlineProcessor(WIKILINK_RE, self.getConfigs())
        wikilinkPattern.md = md
        md.inlinePatterns.register(wikilinkPattern, 'wikilink', 75)