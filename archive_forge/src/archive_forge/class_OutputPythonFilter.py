from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
class OutputPythonFilter(OutputFilter):

    def _process(self, stream, varname, has_nl):
        if self.count > 1:
            yield sql.Token(T.Whitespace, '\n')
        yield sql.Token(T.Name, varname)
        yield sql.Token(T.Whitespace, ' ')
        yield sql.Token(T.Operator, '=')
        yield sql.Token(T.Whitespace, ' ')
        if has_nl:
            yield sql.Token(T.Operator, '(')
        yield sql.Token(T.Text, "'")
        for token in stream:
            if token.is_whitespace and '\n' in token.value:
                yield sql.Token(T.Text, " '")
                yield sql.Token(T.Whitespace, '\n')
                yield sql.Token(T.Whitespace, ' ' * (len(varname) + 4))
                yield sql.Token(T.Text, "'")
                after_lb = token.value.split('\n', 1)[1]
                if after_lb:
                    yield sql.Token(T.Whitespace, after_lb)
                continue
            elif "'" in token.value:
                token.value = token.value.replace("'", "\\'")
            yield sql.Token(T.Text, token.value)
        yield sql.Token(T.Text, "'")
        if has_nl:
            yield sql.Token(T.Operator, ')')