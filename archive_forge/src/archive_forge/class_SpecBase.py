import asyncio
import enum
import json
import pathlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import (
from traitlets import Any as Any_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable
class SpecBase:
    """Base for a spec finder that returns a spec for starting a language server"""
    key = ''
    languages: List[Text] = []
    args: List[Token] = []
    spec: LanguageServerSpec = {}

    def is_installed(self, mgr: LanguageServerManagerAPI) -> bool:
        """Whether the language server is installed or not.

        This method may become abstract in the next major release."""
        return True

    def __call__(self, mgr: LanguageServerManagerAPI) -> KeyedLanguageServerSpecs:
        return {}