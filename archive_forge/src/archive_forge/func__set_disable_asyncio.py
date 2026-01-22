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
@pre
def _set_disable_asyncio(opt, file_config):
    if opt.disable_asyncio:
        asyncio.ENABLE_ASYNCIO = False