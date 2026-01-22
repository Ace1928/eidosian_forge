import logging
import optparse
import os
import re
import shlex
import urllib.parse
from optparse import Values
from typing import (
from pip._internal.cli import cmdoptions
from pip._internal.exceptions import InstallationError, RequirementsFileParseError
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.encoding import auto_decode
from pip._internal.utils.urls import get_url_scheme
def _parse_and_recurse(self, filename: str, constraint: bool) -> Generator[ParsedLine, None, None]:
    for line in self._parse_file(filename, constraint):
        if not line.is_requirement and (line.opts.requirements or line.opts.constraints):
            if line.opts.requirements:
                req_path = line.opts.requirements[0]
                nested_constraint = False
            else:
                req_path = line.opts.constraints[0]
                nested_constraint = True
            if SCHEME_RE.search(filename):
                req_path = urllib.parse.urljoin(filename, req_path)
            elif not SCHEME_RE.search(req_path):
                req_path = os.path.join(os.path.dirname(filename), req_path)
            yield from self._parse_and_recurse(req_path, nested_constraint)
        else:
            yield line