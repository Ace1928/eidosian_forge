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
@post
def __ensure_cext(opt, file_config):
    if os.environ.get('REQUIRE_SQLALCHEMY_CEXT', '0') == '1':
        from sqlalchemy.util import has_compiled_ext
        try:
            has_compiled_ext(raise_=True)
        except ImportError as err:
            raise AssertionError("REQUIRE_SQLALCHEMY_CEXT is set but can't import the cython extensions") from err