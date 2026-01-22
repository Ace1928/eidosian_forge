from datetime import datetime
import functools
import os
import platform
import re
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import Optional
from typing import Tuple
from typing import Union
import xml.etree.ElementTree as ET
from _pytest import nodes
from _pytest import timing
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprFileLocation
from _pytest.config import Config
from _pytest.config import filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
def _warn_incompatibility_with_xunit2(request: FixtureRequest, fixture_name: str) -> None:
    """Emit a PytestWarning about the given fixture being incompatible with newer xunit revisions."""
    from _pytest.warning_types import PytestWarning
    xml = request.config.stash.get(xml_key, None)
    if xml is not None and xml.family not in ('xunit1', 'legacy'):
        request.node.warn(PytestWarning(f"{fixture_name} is incompatible with junit_family '{xml.family}' (use 'legacy' or 'xunit1')"))