import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
@classmethod
def _from_jsonschema(cls, ex: JsonSchemaValueException):
    formatter = _ErrorFormatting(ex)
    obj = cls(str(formatter), ex.value, formatter.name, ex.definition, ex.rule)
    debug_code = os.getenv('JSONSCHEMA_DEBUG_CODE_GENERATION', 'false').lower()
    if debug_code != 'false':
        obj.__cause__, obj.__traceback__ = (ex.__cause__, ex.__traceback__)
    obj._original_message = ex.message
    obj.summary = formatter.summary
    obj.details = formatter.details
    return obj