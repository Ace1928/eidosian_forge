import sys
import traceback
from mako import compat
from mako import util
def _install_pygments():
    global syntax_highlight, pygments_html_formatter
    from mako.ext.pygmentplugin import syntax_highlight
    from mako.ext.pygmentplugin import pygments_html_formatter