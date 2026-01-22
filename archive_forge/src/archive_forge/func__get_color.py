import sys
from pygments.formatter import Formatter
from pygments.token import Keyword, Name, Comment, String, Error, \
from pygments.console import ansiformat
from pygments.util import get_choice_opt
def _get_color(self, ttype):
    colors = self.colorscheme.get(ttype)
    while colors is None:
        ttype = ttype.parent
        colors = self.colorscheme.get(ttype)
    return colors[self.darkbg]