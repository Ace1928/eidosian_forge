from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def valid_final(token):
    return token is not None and token.match(*sql.TypedLiteral.M_EXTEND)