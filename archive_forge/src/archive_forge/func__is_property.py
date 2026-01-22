import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _is_property(self, path: Sequence[str]):
    """Check if the given path can correspond to an arbitrarily named property"""
    counter = 0
    for key in path[-2::-1]:
        if key not in {'properties', 'patternProperties'}:
            break
        counter += 1
    return counter % 2 == 1