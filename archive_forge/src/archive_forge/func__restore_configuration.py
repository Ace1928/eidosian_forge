import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
@classmethod
def _restore_configuration(cls, saved):
    base = cls.configurable_base()
    base.__impl_class = saved[0]
    base.__impl_kwargs = saved[1]