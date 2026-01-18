import contextlib
import json
import re
from urllib import parse as urlparse
from urllib.parse import unquote
from .exceptions import JsonSchemaDefinitionException

        Walk thru schema and dereferencing ``id`` and ``$ref`` instances
        