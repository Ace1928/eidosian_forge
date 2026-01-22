import re
from pygments.lexers import (
from pygments.lexer import (
from pygments.token import (
from pygments.util import get_bool_opt
class IPythonConsoleLexer(Lexer):
    """
    An IPython console lexer for IPython code-blocks and doctests, such as:

    .. code-block:: rst

        .. code-block:: ipythonconsole

            In [1]: a = 'foo'

            In [2]: a
            Out[2]: 'foo'

            In [3]: print(a)
            foo


    Support is also provided for IPython exceptions:

    .. code-block:: rst

        .. code-block:: ipythonconsole

            In [1]: raise Exception
            Traceback (most recent call last):
            ...
            Exception

    """
    name = 'IPython console session'
    aliases = ['ipythonconsole']
    mimetypes = ['text/x-ipython-console']
    in1_regex = 'In \\[[0-9]+\\]: '
    in2_regex = '   \\.\\.+\\.: '
    out_regex = 'Out\\[[0-9]+\\]: '
    ipytb_start = re.compile('^(\\^C)?(-+\\n)|^(  File)(.*)(, line )(\\d+\\n)')

    def __init__(self, **options):
        """Initialize the IPython console lexer.

        Parameters
        ----------
        python3 : bool
            If `True`, then the console inputs are parsed using a Python 3
            lexer. Otherwise, they are parsed using a Python 2 lexer.
        in1_regex : RegexObject
            The compiled regular expression used to detect the start
            of inputs. Although the IPython configuration setting may have a
            trailing whitespace, do not include it in the regex. If `None`,
            then the default input prompt is assumed.
        in2_regex : RegexObject
            The compiled regular expression used to detect the continuation
            of inputs. Although the IPython configuration setting may have a
            trailing whitespace, do not include it in the regex. If `None`,
            then the default input prompt is assumed.
        out_regex : RegexObject
            The compiled regular expression used to detect outputs. If `None`,
            then the default output prompt is assumed.

        """
        self.python3 = get_bool_opt(options, 'python3', False)
        if self.python3:
            self.aliases = ['ipython3console']
        else:
            self.aliases = ['ipython2console', 'ipythonconsole']
        in1_regex = options.get('in1_regex', self.in1_regex)
        in2_regex = options.get('in2_regex', self.in2_regex)
        out_regex = options.get('out_regex', self.out_regex)
        in1_regex_rstrip = in1_regex.rstrip() + '\n'
        in2_regex_rstrip = in2_regex.rstrip() + '\n'
        out_regex_rstrip = out_regex.rstrip() + '\n'
        attrs = ['in1_regex', 'in2_regex', 'out_regex', 'in1_regex_rstrip', 'in2_regex_rstrip', 'out_regex_rstrip']
        for attr in attrs:
            self.__setattr__(attr, re.compile(locals()[attr]))
        Lexer.__init__(self, **options)
        if self.python3:
            pylexer = IPython3Lexer
            tblexer = IPythonTracebackLexer
        else:
            pylexer = IPythonLexer
            tblexer = IPythonTracebackLexer
        self.pylexer = pylexer(**options)
        self.tblexer = tblexer(**options)
        self.reset()

    def reset(self):
        self.mode = 'output'
        self.index = 0
        self.buffer = u''
        self.insertions = []

    def buffered_tokens(self):
        """
        Generator of unprocessed tokens after doing insertions and before
        changing to a new state.

        """
        if self.mode == 'output':
            tokens = [(0, Generic.Output, self.buffer)]
        elif self.mode == 'input':
            tokens = self.pylexer.get_tokens_unprocessed(self.buffer)
        else:
            tokens = self.tblexer.get_tokens_unprocessed(self.buffer)
        for i, t, v in do_insertions(self.insertions, tokens):
            yield (self.index + i, t, v)
        self.index += len(self.buffer)
        self.buffer = u''
        self.insertions = []

    def get_mci(self, line):
        """
        Parses the line and returns a 3-tuple: (mode, code, insertion).

        `mode` is the next mode (or state) of the lexer, and is always equal
        to 'input', 'output', or 'tb'.

        `code` is a portion of the line that should be added to the buffer
        corresponding to the next mode and eventually lexed by another lexer.
        For example, `code` could be Python code if `mode` were 'input'.

        `insertion` is a 3-tuple (index, token, text) representing an
        unprocessed "token" that will be inserted into the stream of tokens
        that are created from the buffer once we change modes. This is usually
        the input or output prompt.

        In general, the next mode depends on current mode and on the contents
        of `line`.

        """
        in2_match = self.in2_regex.match(line)
        in2_match_rstrip = self.in2_regex_rstrip.match(line)
        if in2_match and in2_match.group().rstrip() == line.rstrip() or in2_match_rstrip:
            end_input = True
        else:
            end_input = False
        if end_input and self.mode != 'tb':
            mode = 'output'
            code = u''
            insertion = (0, Generic.Prompt, line)
            return (mode, code, insertion)
        out_match = self.out_regex.match(line)
        out_match_rstrip = self.out_regex_rstrip.match(line)
        if out_match or out_match_rstrip:
            mode = 'output'
            if out_match:
                idx = out_match.end()
            else:
                idx = out_match_rstrip.end()
            code = line[idx:]
            insertion = (0, Generic.Heading, line[:idx])
            return (mode, code, insertion)
        in1_match = self.in1_regex.match(line)
        if in1_match or (in2_match and self.mode != 'tb'):
            mode = 'input'
            if in1_match:
                idx = in1_match.end()
            else:
                idx = in2_match.end()
            code = line[idx:]
            insertion = (0, Generic.Prompt, line[:idx])
            return (mode, code, insertion)
        in1_match_rstrip = self.in1_regex_rstrip.match(line)
        if in1_match_rstrip or (in2_match_rstrip and self.mode != 'tb'):
            mode = 'input'
            if in1_match_rstrip:
                idx = in1_match_rstrip.end()
            else:
                idx = in2_match_rstrip.end()
            code = line[idx:]
            insertion = (0, Generic.Prompt, line[:idx])
            return (mode, code, insertion)
        if self.ipytb_start.match(line):
            mode = 'tb'
            code = line
            insertion = None
            return (mode, code, insertion)
        if self.mode in ('input', 'output'):
            mode = 'output'
        else:
            mode = 'tb'
        code = line
        insertion = None
        return (mode, code, insertion)

    def get_tokens_unprocessed(self, text):
        self.reset()
        for match in line_re.finditer(text):
            line = match.group()
            mode, code, insertion = self.get_mci(line)
            if mode != self.mode:
                for token in self.buffered_tokens():
                    yield token
                self.mode = mode
            if insertion:
                self.insertions.append((len(self.buffer), [insertion]))
            self.buffer += code
        for token in self.buffered_tokens():
            yield token