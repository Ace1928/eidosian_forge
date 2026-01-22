import os
from functools import wraps
def add_py3_data(path):
    for item in _PY3_DATA_UPDATES:
        if item in str(path) and '/PY3' not in str(path):
            pos = path.index(item) + len(item)
            if path[pos:pos + 4] == '.zip':
                pos += 4
            path = path[:pos] + '/PY3' + path[pos:]
            break
    return path