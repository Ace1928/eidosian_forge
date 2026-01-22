import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, words
from pygments.token import Punctuation, Whitespace, Error, \
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pygments.util import iteritems
from pygments.lexers._postgres_builtins import KEYWORDS, DATATYPES, \
from pygments.lexers import _tsql_builtins
class PostgresBase(object):
    """Base class for Postgres-related lexers.

    This is implemented as a mixin to avoid the Lexer metaclass kicking in.
    this way the different lexer don't have a common Lexer ancestor. If they
    had, _tokens could be created on this ancestor and not updated for the
    other classes, resulting e.g. in PL/pgSQL parsed as SQL. This shortcoming
    seem to suggest that regexp lexers are not really subclassable.
    """

    def get_tokens_unprocessed(self, text, *args):
        self.text = text
        for x in super(PostgresBase, self).get_tokens_unprocessed(text, *args):
            yield x

    def _get_lexer(self, lang):
        if lang.lower() == 'sql':
            return get_lexer_by_name('postgresql', **self.options)
        tries = [lang]
        if lang.startswith('pl'):
            tries.append(lang[2:])
        if lang.endswith('u'):
            tries.append(lang[:-1])
        if lang.startswith('pl') and lang.endswith('u'):
            tries.append(lang[2:-1])
        for l in tries:
            try:
                return get_lexer_by_name(l, **self.options)
            except ClassNotFound:
                pass
        else:
            return None