import re
from sqlparse import tokens as T
from sqlparse.exceptions import SQLParseError
from sqlparse.utils import imt, remove_quotes
class TypedLiteral(TokenList):
    """A typed literal, such as "date '2001-09-28'" or "interval '2 hours'"."""
    M_OPEN = [(T.Name.Builtin, None), (T.Keyword, 'TIMESTAMP')]
    M_CLOSE = (T.String.Single, None)
    M_EXTEND = (T.Keyword, ('DAY', 'HOUR', 'MINUTE', 'MONTH', 'SECOND', 'YEAR'))