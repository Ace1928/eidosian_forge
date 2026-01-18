from __future__ import annotations
import argparse
import ast
import functools
import logging
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple
from flake8 import defaults
from flake8 import utils
from flake8._compat import FSTRING_END
from flake8._compat import FSTRING_MIDDLE
from flake8.plugins.finder import LoadedPlugin
def update_checker_state_for(self, plugin: LoadedPlugin) -> None:
    """Update the checker_state attribute for the plugin."""
    if 'checker_state' in plugin.parameters:
        self.checker_state = self._checker_states.setdefault(plugin.entry_name, {})