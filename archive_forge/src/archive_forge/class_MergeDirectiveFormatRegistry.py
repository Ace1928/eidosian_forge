import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
class MergeDirectiveFormatRegistry(registry.Registry):

    def register(self, directive, format_string=None):
        if format_string is None:
            format_string = directive._format_string
        registry.Registry.register(self, format_string, directive)