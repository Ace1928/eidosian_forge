import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _draw_text(self, pos, text, font, **kw):
    """
        Remember a single drawable tuple to paint later.
        """
    self.drawables.append((pos, text, font, kw))