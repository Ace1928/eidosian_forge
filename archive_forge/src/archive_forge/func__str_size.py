from typing import Any, Dict, Iterable
def _str_size(x):
    return len(x) if isinstance(x, bytes) else len(x.encode('utf-8'))