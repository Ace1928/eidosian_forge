from within Jinja templates.
from html import escape
from warnings import warn
from traitlets import Dict, observe
from nbconvert.utils.base import NbConvertBase
class Highlight2HTML(NbConvertBase):
    """Convert highlighted code to html."""
    extra_formatter_options = Dict({}, help="\n        Extra set of options to control how code is highlighted.\n\n        Passed through to the pygments' HtmlFormatter class.\n        See available list in https://pygments.org/docs/formatters/#HtmlFormatter\n        ", config=True)

    def __init__(self, pygments_lexer=None, **kwargs):
        """Initialize the converter."""
        self.pygments_lexer = pygments_lexer or 'ipython3'
        super().__init__(**kwargs)

    @observe('default_language')
    def _default_language_changed(self, change):
        warn('Setting default_language in config is deprecated as of 5.0, please use language_info metadata instead.', stacklevel=2)
        self.pygments_lexer = change['new']

    def __call__(self, source, language=None, metadata=None):
        """
        Return a syntax-highlighted version of the input source as html output.

        Parameters
        ----------
        source : str
            source of the cell to highlight
        language : str
            language to highlight the syntax of
        metadata : NotebookNode cell metadata
            metadata of the cell to highlight
        """
        from pygments.formatters import HtmlFormatter
        if not language:
            language = self.pygments_lexer
        return _pygments_highlight(source if len(source) > 0 else ' ', HtmlFormatter(cssclass=escape(f' highlight hl-{language}'), **self.extra_formatter_options), language, metadata)