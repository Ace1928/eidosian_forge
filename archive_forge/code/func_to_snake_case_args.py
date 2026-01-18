import typing
from re import sub
def to_snake_case_args(text: str):
    """
    multiMaster -> multi-master
    """
    return '-'.join(sub('([A-Z][a-z]+)', ' \\1', sub('([A-Z]+)', ' \\1', text.replace('-', ' '))).split()).lower()