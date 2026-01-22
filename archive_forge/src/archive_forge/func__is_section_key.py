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
def _is_section_key(key: str) -> bool:
    """Is a Config key a section name (does it start with a capital)?"""
    return bool(key and key[0].upper() == key[0] and (not key.startswith('_')))