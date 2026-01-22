import six
import subprocess
from ebooklib.plugins.base import BasePlugin
from ebooklib.utils import parse_html_string
class TidyPlugin(BasePlugin):
    NAME = 'Tidy HTML'
    OPTIONS = {'char-encoding': 'utf8', 'tidy-mark': 'no'}

    def __init__(self, extra={}):
        self.options = dict(self.OPTIONS)
        self.options.update(extra)

    def html_before_write(self, book, chapter):
        if not chapter.content:
            return None
        _, chapter.content = tidy_cleanup(chapter.content, **self.options)
        return chapter.content

    def html_after_read(self, book, chapter):
        if not chapter.content:
            return None
        _, chapter.content = tidy_cleanup(chapter.content, **self.options)
        return chapter.content