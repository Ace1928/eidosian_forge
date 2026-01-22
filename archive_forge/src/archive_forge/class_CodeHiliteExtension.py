from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue
from typing import TYPE_CHECKING, Callable, Any
class CodeHiliteExtension(Extension):
    """ Add source code highlighting to markdown code blocks. """

    def __init__(self, **kwargs):
        self.config = {'linenums': [None, 'Use lines numbers. True|table|inline=yes, False=no, None=auto. Default: `None`.'], 'guess_lang': [True, 'Automatic language detection - Default: `True`.'], 'css_class': ['codehilite', 'Set class name for wrapper <div> - Default: `codehilite`.'], 'pygments_style': ['default', 'Pygments HTML Formatter Style (Colorscheme). Default: `default`.'], 'noclasses': [False, 'Use inline styles instead of CSS classes - Default `False`.'], 'use_pygments': [True, 'Highlight code blocks with pygments. Disable if using a JavaScript library. Default: `True`.'], 'lang_prefix': ['language-', 'Prefix prepended to the language when `use_pygments` is false. Default: `language-`.'], 'pygments_formatter': ['html', 'Use a specific formatter for Pygments highlighting. Default: `html`.']}
        ' Default configuration options. '
        for key, value in kwargs.items():
            if key in self.config:
                self.setConfig(key, value)
            else:
                if isinstance(value, str):
                    try:
                        value = parseBoolValue(value, preserve_none=True)
                    except ValueError:
                        pass
                self.config[key] = [value, '']

    def extendMarkdown(self, md):
        """ Add `HilitePostprocessor` to Markdown instance. """
        hiliter = HiliteTreeprocessor(md)
        hiliter.config = self.getConfigs()
        md.treeprocessors.register(hiliter, 'hilite', 30)
        md.registerExtension(self)