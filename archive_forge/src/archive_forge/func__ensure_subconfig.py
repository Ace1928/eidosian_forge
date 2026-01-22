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
def _ensure_subconfig(self) -> None:
    """ensure that sub-dicts that should be Config objects are

        casts dicts that are under section keys to Config objects,
        which is necessary for constructing Config objects from dict literals.
        """
    for key in self:
        obj = self[key]
        if _is_section_key(key) and isinstance(obj, dict) and (not isinstance(obj, Config)):
            setattr(self, key, Config(obj))