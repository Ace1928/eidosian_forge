from __future__ import annotations
import functools
import json
import logging
import os
import pprint
import re
import sys
import typing as t
from collections import OrderedDict, defaultdict
from contextlib import suppress
from copy import deepcopy
from logging.config import dictConfig
from textwrap import dedent
from traitlets.config.configurable import Configurable, SingletonConfigurable
from traitlets.config.loader import (
from traitlets.traitlets import (
from traitlets.utils.bunch import Bunch
from traitlets.utils.nested_update import nested_update
from traitlets.utils.text import indent, wrap_paragraphs
from ..utils import cast_unicode
from ..utils.importstring import import_item
@classmethod
def _handle_argcomplete_for_subcommand(cls) -> None:
    """Helper for `argcomplete` to recognize `traitlets` subcommands

        `argcomplete` does not know that `traitlets` has already consumed subcommands,
        as it only "sees" the final `argparse.ArgumentParser` that is constructed.
        (Indeed `KVArgParseConfigLoader` does not get passed subcommands at all currently.)
        We explicitly manipulate the environment variables used internally by `argcomplete`
        to get it to skip over the subcommand tokens.
        """
    if '_ARGCOMPLETE' not in os.environ:
        return
    try:
        from traitlets.config.argcomplete_config import increment_argcomplete_index
        increment_argcomplete_index()
    except (ImportError, ModuleNotFoundError):
        pass