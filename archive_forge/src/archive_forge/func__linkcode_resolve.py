import glob
import importlib
import inspect
import logging
import os
import re
import sys
from typing import Iterable, List, Optional, Tuple, Union
def _linkcode_resolve(domain: str, github_user: str, github_repo: str, info: dict) -> str:

    def find_source() -> Tuple[str, int, int]:
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        fname = str(inspect.getsourcefile(obj))
        if any((s in fname for s in ('readthedocs', 'rtfd', 'checkouts'))):
            path_top = os.path.abspath(os.path.join('..', '..', '..'))
            fname = str(os.path.relpath(fname, start=path_top))
        else:
            fname = f'master/{os.path.relpath(fname, start=os.path.abspath('..'))}'
        source, line_start = inspect.getsourcelines(obj)
        return (fname, line_start, line_start + len(source) - 1)
    if domain != 'py' or not info['module']:
        return ''
    try:
        filename = '%s#L%d-L%d' % find_source()
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'
    branch = filename.split('/')[0]
    branch = {'latest': 'master', 'stable': 'master'}.get(branch, branch)
    filename = '/'.join([branch] + filename.split('/')[1:])
    return f'https://github.com/{github_user}/{github_repo}/blob/{filename}'