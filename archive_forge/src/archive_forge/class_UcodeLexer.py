import re
from pygments.lexer import RegexLexer, include, bygroups, words, using, this
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class UcodeLexer(RegexLexer):
    """
    Lexer for Icon ucode files.

    .. versionadded:: 2.4
    """
    name = 'ucode'
    aliases = ['ucode']
    filenames = ['*.u', '*.u1', '*.u2']
    mimetypes = []
    flags = re.MULTILINE
    tokens = {'root': [('(#.*\\n)', Comment), (words(('con', 'declend', 'end', 'global', 'impl', 'invocable', 'lab', 'link', 'local', 'record', 'uid', 'unions', 'version'), prefix='\\b', suffix='\\b'), Name.Function), (words(('colm', 'filen', 'line', 'synt'), prefix='\\b', suffix='\\b'), Comment), (words(('asgn', 'bang', 'bscan', 'cat', 'ccase', 'chfail', 'coact', 'cofail', 'compl', 'coret', 'create', 'cset', 'diff', 'div', 'dup', 'efail', 'einit', 'end', 'eqv', 'eret', 'error', 'escan', 'esusp', 'field', 'goto', 'init', 'int', 'inter', 'invoke', 'keywd', 'lconcat', 'lexeq', 'lexge', 'lexgt', 'lexle', 'lexlt', 'lexne', 'limit', 'llist', 'lsusp', 'mark', 'mark0', 'minus', 'mod', 'mult', 'neg', 'neqv', 'nonnull', 'noop', 'null', 'number', 'numeq', 'numge', 'numgt', 'numle', 'numlt', 'numne', 'pfail', 'plus', 'pnull', 'pop', 'power', 'pret', 'proc', 'psusp', 'push1', 'pushn1', 'random', 'rasgn', 'rcv', 'rcvbk', 'real', 'refresh', 'rswap', 'sdup', 'sect', 'size', 'snd', 'sndbk', 'str', 'subsc', 'swap', 'tabmat', 'tally', 'toby', 'trace', 'unmark', 'value', 'var'), prefix='\\b', suffix='\\b'), Keyword.Declaration), (words(('any', 'case', 'endcase', 'endevery', 'endif', 'endifelse', 'endrepeat', 'endsuspend', 'enduntil', 'endwhile', 'every', 'if', 'ifelse', 'repeat', 'suspend', 'until', 'while'), prefix='\\b', suffix='\\b'), Name.Constant), ('\\d+(\\s*|\\.$|$)', Number.Integer), ('[+-]?\\d*\\.\\d+(E[-+]?\\d+)?', Number.Float), ('[+-]?\\d+\\.\\d*(E[-+]?\\d+)?', Number.Float), ("(<>|=>|[()|:;,.'`]|[{}]|[%^]|[&?])", Punctuation), ('\\s+\\b', Text), ('[\\w-]+', Text)]}

    def analyse_text(text):
        """endsuspend and endrepeat are unique to this language, and
        \\self, /self doesn't seem to get used anywhere else either."""
        result = 0
        if 'endsuspend' in text:
            result += 0.1
        if 'endrepeat' in text:
            result += 0.1
        if ':=' in text:
            result += 0.01
        if 'procedure' in text and 'end' in text:
            result += 0.01
        if '\\self' in text and '/self' in text:
            result += 0.5
        return result