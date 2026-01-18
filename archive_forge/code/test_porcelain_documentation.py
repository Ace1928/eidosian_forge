import os
import platform
import sys
from unittest import skipIf
from dulwich import porcelain
from ..test_porcelain import PorcelainGpgTestCase
from ..utils import build_commit_graph
from .utils import CompatTestCase, run_git_or_fail
Compatibility tests for dulwich.porcelain.