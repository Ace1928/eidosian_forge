from __future__ import annotations
import argparse
import copy
import functools
import json
import os
import re
import sys
import typing as t
from logging import Logger
from traitlets.traitlets import Any, Container, Dict, HasTraits, List, TraitType, Undefined
from ..utils import cast_unicode, filefind, warnings
def _argcomplete(self, classes: list[t.Any], subcommands: SubcommandsDict | None) -> None:
    """If argcomplete is enabled, allow triggering command-line autocompletion"""
    try:
        import argcomplete
    except ImportError:
        return
    from . import argcomplete_config
    finder = argcomplete_config.ExtendedCompletionFinder()
    finder.config_classes = classes
    finder.subcommands = list(subcommands or [])
    finder(self.parser, **getattr(self, '_argcomplete_kwargs', {}))