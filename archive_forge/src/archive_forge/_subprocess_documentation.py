import os
import subprocess
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import IO, Generator, List, Optional, Tuple, Union
from .logging import get_logger
Run a subprocess in an interactive mode in a context manager.

    Args:
        command (`str` or `List[str]`):
            The command to execute as a string or list of strings.
        folder (`str`, *optional*):
            The folder in which to run the command. Defaults to current working
            directory (from `os.getcwd()`).
        kwargs (`Dict[str]`):
            Keyword arguments to be passed to the `subprocess.run` underlying command.

    Returns:
        `Tuple[IO[str], IO[str]]`: A tuple with `stdin` and `stdout` to interact
        with the process (input and output are utf-8 encoded).

    Example:
    ```python
    with _interactive_subprocess("git credential-store get") as (stdin, stdout):
        # Write to stdin
        stdin.write("url=hf.co
username=obama
".encode("utf-8"))
        stdin.flush()

        # Read from stdout
        output = stdout.read().decode("utf-8")
    ```
    