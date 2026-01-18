import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_stringio(self) -> None:
    with sinks.BufferFixture() as out:
        ap = autopage.AutoPager(out.stream)
        self.assertFalse(ap.to_terminal())