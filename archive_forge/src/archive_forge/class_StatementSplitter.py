from sqlparse import sql, tokens as T
class StatementSplitter(object):
    """Filter that split stream at individual statements"""

    def __init__(self):
        self._reset()

    def _reset(self):
        """Set the filter attributes to its default values"""
        self._in_declare = False
        self._is_create = False
        self._begin_depth = 0
        self.consume_ws = False
        self.tokens = []
        self.level = 0

    def _change_splitlevel(self, ttype, value):
        """Get the new split level (increase, decrease or remain equal)"""
        if ttype not in T.Keyword:
            return 0
        unified = value.upper()
        if ttype is T.Keyword.DDL and unified.startswith('CREATE'):
            self._is_create = True
            return 0
        if unified == 'DECLARE' and self._is_create and (self._begin_depth == 0):
            self._in_declare = True
            return 1
        if unified == 'BEGIN':
            self._begin_depth += 1
            if self._is_create:
                return 1
            return 0
        if unified == 'END':
            self._begin_depth = max(0, self._begin_depth - 1)
            return -1
        if unified in ('IF', 'FOR', 'WHILE') and self._is_create and (self._begin_depth > 0):
            return 1
        if unified in ('END IF', 'END FOR', 'END WHILE'):
            return -1
        return 0

    def process(self, stream):
        """Process the stream"""
        EOS_TTYPE = (T.Whitespace, T.Comment.Single)
        for ttype, value in stream:
            if self.consume_ws and ttype not in EOS_TTYPE:
                yield sql.Statement(self.tokens)
                self._reset()
            self.level += self._change_splitlevel(ttype, value)
            self.tokens.append(sql.Token(ttype, value))
            if self.level <= 0 and ttype is T.Punctuation and (value == ';'):
                self.consume_ws = True
        if self.tokens:
            yield sql.Statement(self.tokens)