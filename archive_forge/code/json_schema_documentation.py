import inspect
import json
import re
from typing import Callable, Optional
from jsonschema.protocols import Validator
from pydantic import create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012
Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    