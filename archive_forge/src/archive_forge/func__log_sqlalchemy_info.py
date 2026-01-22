from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def _log_sqlalchemy_info(session):
    import sqlalchemy
    from sqlalchemy import __version__
    from sqlalchemy.util import has_compiled_ext
    from sqlalchemy.util._has_cy import _CYEXTENSION_MSG
    greet = 'sqlalchemy installation'
    site = 'no user site' if sys.flags.no_user_site else 'user site loaded'
    msgs = [f'SQLAlchemy {__version__} ({site})', f'Path: {sqlalchemy.__file__}']
    if has_compiled_ext():
        from sqlalchemy.cyextension import util
        msgs.append(f'compiled extension enabled, e.g. {util.__file__} ')
    else:
        msgs.append(f'compiled extension not enabled; {_CYEXTENSION_MSG}')
    pm = session.config.pluginmanager.get_plugin('terminalreporter')
    if pm:
        pm.write_sep('=', greet)
        for m in msgs:
            pm.write_line(m)
    else:
        print('=' * 25, greet, '=' * 25)
        for m in msgs:
            print(m)