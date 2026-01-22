import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _inline_attrs(self, schema: dict, path: Sequence[str]) -> Iterator[str]:
    for key, value in schema.items():
        child_path = [*path, key]
        yield f'{self._label(child_path)}: {self._value(value, child_path)}'