from __future__ import annotations
import codecs
import json
import locale
import os
import platform
import struct
import sys
from typing import TYPE_CHECKING
from pandas.compat._optional import (
def _get_dependency_info() -> dict[str, JSONSerializable]:
    """
    Returns dependency information as a JSON serializable dictionary.
    """
    deps = ['pandas', 'numpy', 'pytz', 'dateutil', 'setuptools', 'pip', 'Cython', 'pytest', 'hypothesis', 'sphinx', 'blosc', 'feather', 'xlsxwriter', 'lxml.etree', 'html5lib', 'pymysql', 'psycopg2', 'jinja2', 'IPython', 'pandas_datareader']
    deps.extend(list(VERSIONS))
    result: dict[str, JSONSerializable] = {}
    for modname in deps:
        mod = import_optional_dependency(modname, errors='ignore')
        result[modname] = get_version(mod) if mod else None
    return result