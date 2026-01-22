import asyncio
import os
import sys
from IPython.core.debugger import Pdb
from IPython.core.completer import IPCompleter
from .ptutils import IPythonPTCompleter
from .shortcuts import create_ipython_shortcuts
from . import embed
from pathlib import Path
from pygments.token import Token
from prompt_toolkit.application import create_app_session
from prompt_toolkit.shortcuts.prompt import PromptSession
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import InMemoryHistory, FileHistory
from concurrent.futures import ThreadPoolExecutor
from prompt_toolkit import __version__ as ptk_version
def _prompt(self):
    """
        In case other prompt_toolkit apps have to run in parallel to this one (e.g. in madbg),
        create_app_session must be used to prevent mixing up between them. According to the prompt_toolkit docs:

        > If you need multiple applications running at the same time, you have to create a separate
        > `AppSession` using a `with create_app_session():` block.
        """
    with create_app_session():
        return self.pt_app.prompt()