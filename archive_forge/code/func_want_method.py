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
def want_method(cls, fn):
    if not fn.__name__.startswith('test_'):
        return False
    elif fn.__module__ is None:
        return False
    else:
        return True