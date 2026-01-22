from __future__ import annotations
import os
import tempfile
import textwrap
from functools import lru_cache
class CppCompileError(RuntimeError):

    def __init__(self, cmd: list[str], output: str):
        if isinstance(output, bytes):
            output = output.decode('utf-8')
        super().__init__(textwrap.dedent('\n                    C++ compile error\n\n                    Command:\n                    {cmd}\n\n                    Output:\n                    {output}\n                ').strip().format(cmd=' '.join(cmd), output=output))