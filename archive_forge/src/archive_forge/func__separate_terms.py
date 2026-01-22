import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _separate_terms(word: str) -> List[str]:
    """
    >>> _separate_terms("FooBar-foo")
    ['foo', 'bar', 'foo']
    """
    return [w.lower() for w in _CAMEL_CASE_SPLITTER.split(word) if w]