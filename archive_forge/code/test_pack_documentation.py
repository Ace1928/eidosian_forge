import binascii
import os
import re
import shutil
import tempfile
from dulwich.tests import SkipTest
from ...objects import Blob
from ...pack import write_pack
from ..test_pack import PackTests, a_sha, pack1_sha
from .utils import require_git_version, run_git_or_fail
Compatibility tests for reading and writing pack files.