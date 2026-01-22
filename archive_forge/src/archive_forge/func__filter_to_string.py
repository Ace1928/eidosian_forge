from ._base import *
@staticmethod
def _filter_to_string(name: str, f: Filter) -> str:
    if type(f) == str:
        return f'{name}:{f}'
    else:
        return '(' + ' OR '.join((f'{name}:{clause}' for clause in f)) + ')'