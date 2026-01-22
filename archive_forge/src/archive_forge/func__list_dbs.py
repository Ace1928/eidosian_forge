from __future__ import annotations
import abc
from argparse import Namespace
import configparser
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any
from sqlalchemy.testing import asyncio
def _list_dbs(*args):
    if file_config is None:
        read_config(Path.cwd())
    print('Available --db options (use --dburi to override)')
    for macro in sorted(file_config.options('db')):
        print('%20s\t%s' % (macro, file_config.get('db', macro)))
    sys.exit(0)