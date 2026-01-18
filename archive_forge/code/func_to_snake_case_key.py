import typing
from re import sub
def to_snake_case_key(text: str):
    """
    multi_master -> multi-master
    """
    return '-'.join(sub('([A-Z][a-z]+)', ' \\1', sub('([A-Z]+)', ' \\1', text.replace('_', ' '))).split()).lower()