from __future__ import print_function
import sys
import getopt
from textwrap import dedent
from pygments import __version__, highlight
from pygments.util import ClassNotFound, OptionError, docstring_headline, \
from pygments.lexers import get_all_lexers, get_lexer_by_name, guess_lexer, \
from pygments.lexers.special import TextLexer
from pygments.formatters.latex import LatexEmbeddedLexer, LatexFormatter
from pygments.formatters import get_all_formatters, get_formatter_by_name, \
from pygments.formatters.terminal import TerminalFormatter
from pygments.filters import get_all_filters, find_filter_class
from pygments.styles import get_all_styles, get_style_by_name
def _print_list(what):
    if what == 'lexer':
        print()
        print('Lexers:')
        print('~~~~~~~')
        info = []
        for fullname, names, exts, _ in get_all_lexers():
            tup = (', '.join(names) + ':', fullname, exts and '(filenames ' + ', '.join(exts) + ')' or '')
            info.append(tup)
        info.sort()
        for i in info:
            print('* %s\n    %s %s' % i)
    elif what == 'formatter':
        print()
        print('Formatters:')
        print('~~~~~~~~~~~')
        info = []
        for cls in get_all_formatters():
            doc = docstring_headline(cls)
            tup = (', '.join(cls.aliases) + ':', doc, cls.filenames and '(filenames ' + ', '.join(cls.filenames) + ')' or '')
            info.append(tup)
        info.sort()
        for i in info:
            print('* %s\n    %s %s' % i)
    elif what == 'filter':
        print()
        print('Filters:')
        print('~~~~~~~~')
        for name in get_all_filters():
            cls = find_filter_class(name)
            print('* ' + name + ':')
            print('    %s' % docstring_headline(cls))
    elif what == 'style':
        print()
        print('Styles:')
        print('~~~~~~~')
        for name in get_all_styles():
            cls = get_style_by_name(name)
            print('* ' + name + ':')
            print('    %s' % docstring_headline(cls))