import asyncio
import logging
from concurrent.futures import Executor, ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
from multiprocessing import freeze_support
from typing import Set, Tuple
import click
import black
from _black_version import version as __version__
from black.concurrency import maybe_install_uvloop
class InvalidVariantHeader(Exception):
    pass