import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _child_prefix(self, parent_prefix: str, child_prefix: str) -> str:
    return len(parent_prefix) * ' ' + child_prefix