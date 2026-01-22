import asyncio
import itertools
import logging
import subprocess
import textwrap
import types
from typing import List
@staticmethod
def _get_last_n_line(str_data: str, last_n_lines: int) -> str:
    if last_n_lines < 0:
        return str_data
    lines = str_data.strip().split('\n')
    return '\n'.join(lines[-last_n_lines:])