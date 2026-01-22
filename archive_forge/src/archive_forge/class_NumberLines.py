from docutils import ApplicationError
class NumberLines(object):
    """Insert linenumber-tokens at the start of every code line.

    Arguments

       tokens    -- iterable of ``(classes, value)`` tuples
       startline -- first line number
       endline   -- last line number

    Iterating over an instance yields the tokens with a
    ``(['ln'], '<the line number>')`` token added for every code line.
    Multi-line tokens are splitted."""

    def __init__(self, tokens, startline, endline):
        self.tokens = tokens
        self.startline = startline
        self.fmt_str = '%%%dd ' % len(str(endline))

    def __iter__(self):
        lineno = self.startline
        yield (['ln'], self.fmt_str % lineno)
        for ttype, value in self.tokens:
            lines = value.split('\n')
            for line in lines[:-1]:
                yield (ttype, line + '\n')
                lineno += 1
                yield (['ln'], self.fmt_str % lineno)
            yield (ttype, lines[-1])