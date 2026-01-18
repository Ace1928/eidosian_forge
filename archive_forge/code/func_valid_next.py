from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def valid_next(token):
    return True