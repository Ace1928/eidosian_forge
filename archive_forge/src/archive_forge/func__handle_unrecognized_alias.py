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
def _handle_unrecognized_alias(self, arg: str) -> None:
    """Handling for unrecognized alias arguments

        Probably a mistyped alias. By default just log a warning,
        but users can override this to raise an error instead, e.g.
        self.parser.error("Unrecognized alias: '%s'" % arg)
        """
    self.log.warning("Unrecognized alias: '%s', it will have no effect.", arg)