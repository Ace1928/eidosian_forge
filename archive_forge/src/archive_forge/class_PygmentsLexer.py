from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
class PygmentsLexer(Lexer):
    """
    Lexer that calls a pygments lexer.

    Example::

        from pygments.lexers import HtmlLexer
        lexer = PygmentsLexer(HtmlLexer)

    Note: Don't forget to also load a Pygments compatible style. E.g.::

        from prompt_toolkit.styles.from_pygments import style_from_pygments
        from pygments.styles import get_style_by_name
        style = style_from_pygments(get_style_by_name('monokai'))

    :param pygments_lexer_cls: A `Lexer` from Pygments.
    :param sync_from_start: Start lexing at the start of the document. This
        will always give the best results, but it will be slow for bigger
        documents. (When the last part of the document is display, then the
        whole document will be lexed by Pygments on every key stroke.) It is
        recommended to disable this for inputs that are expected to be more
        than 1,000 lines.
    :param syntax_sync: `SyntaxSync` object.
    """
    MIN_LINES_BACKWARDS = 50
    REUSE_GENERATOR_MAX_DISTANCE = 100

    def __init__(self, pygments_lexer_cls, sync_from_start=True, syntax_sync=None):
        assert syntax_sync is None or isinstance(syntax_sync, SyntaxSync)
        self.pygments_lexer_cls = pygments_lexer_cls
        self.sync_from_start = to_cli_filter(sync_from_start)
        self.pygments_lexer = pygments_lexer_cls(stripnl=False, stripall=False, ensurenl=False)
        self.syntax_sync = syntax_sync or RegexSync.from_pygments_lexer_cls(pygments_lexer_cls)

    @classmethod
    def from_filename(cls, filename, sync_from_start=True):
        """
        Create a `Lexer` from a filename.
        """
        from pygments.util import ClassNotFound
        from pygments.lexers import get_lexer_for_filename
        try:
            pygments_lexer = get_lexer_for_filename(filename)
        except ClassNotFound:
            return SimpleLexer()
        else:
            return cls(pygments_lexer.__class__, sync_from_start=sync_from_start)

    def lex_document(self, cli, document):
        """
        Create a lexer function that takes a line number and returns the list
        of (Token, text) tuples as the Pygments lexer returns for that line.
        """
        cache = {}
        line_generators = {}

        def get_syntax_sync():
            """ The Syntax synchronisation objcet that we currently use. """
            if self.sync_from_start(cli):
                return SyncFromStart()
            else:
                return self.syntax_sync

        def find_closest_generator(i):
            """ Return a generator close to line 'i', or None if none was fonud. """
            for generator, lineno in line_generators.items():
                if lineno < i and i - lineno < self.REUSE_GENERATOR_MAX_DISTANCE:
                    return generator

        def create_line_generator(start_lineno, column=0):
            """
            Create a generator that yields the lexed lines.
            Each iteration it yields a (line_number, [(token, text), ...]) tuple.
            """

            def get_tokens():
                text = '\n'.join(document.lines[start_lineno:])[column:]
                for _, t, v in self.pygments_lexer.get_tokens_unprocessed(text):
                    yield (t, v)
            return enumerate(split_lines(get_tokens()), start_lineno)

        def get_generator(i):
            """
            Find an already started generator that is close, or create a new one.
            """
            generator = find_closest_generator(i)
            if generator:
                return generator
            i = max(0, i - self.MIN_LINES_BACKWARDS)
            if i == 0:
                row = 0
                column = 0
            else:
                row, column = get_syntax_sync().get_sync_start_position(document, i)
            generator = find_closest_generator(i)
            if generator:
                return generator
            else:
                generator = create_line_generator(row, column)
            if column:
                next(generator)
                row += 1
            line_generators[generator] = row
            return generator

        def get_line(i):
            """ Return the tokens for a given line number. """
            try:
                return cache[i]
            except KeyError:
                generator = get_generator(i)
                for num, line in generator:
                    cache[num] = line
                    if num == i:
                        line_generators[generator] = i
                        if num + 1 in cache:
                            del cache[num + 1]
                        return cache[num]
            return []
        return get_line