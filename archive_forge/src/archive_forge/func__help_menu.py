import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def _help_menu(self, verbose: bool=False) -> None:
    """Show a list of commands which help can be displayed for"""
    cmds_cats, cmds_doc, cmds_undoc, help_topics = self._build_command_info()
    if not cmds_cats:
        self.poutput(self.doc_leader)
        self._print_topics(self.doc_header, cmds_doc, verbose)
    else:
        self.poutput(self.doc_leader)
        self.poutput(self.doc_header, end='\n\n')
        for category in sorted(cmds_cats.keys(), key=self.default_sort_key):
            self._print_topics(category, cmds_cats[category], verbose)
        self._print_topics(self.default_category, cmds_doc, verbose)
    self.print_topics(self.misc_header, help_topics, 15, 80)
    self.print_topics(self.undoc_header, cmds_undoc, 15, 80)