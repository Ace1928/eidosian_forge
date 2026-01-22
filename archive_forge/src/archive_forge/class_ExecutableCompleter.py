from __future__ import unicode_literals
from prompt_toolkit.completion import Completer, Completion
import os
class ExecutableCompleter(PathCompleter):
    """
    Complete only excutable files in the current path.
    """

    def __init__(self):
        (PathCompleter.__init__(self, only_directories=False, min_input_len=1, get_paths=lambda: os.environ.get('PATH', '').split(os.pathsep), file_filter=lambda name: os.access(name, os.X_OK), expanduser=True),)