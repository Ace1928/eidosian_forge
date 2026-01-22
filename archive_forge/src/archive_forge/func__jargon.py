import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _jargon(self, term: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(term, list):
        return [self.jargon.get(t, t) for t in term]
    return self.jargon.get(term, term)