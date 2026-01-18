import os
import shutil
import stat
import sys
import tempfile
from dulwich import errors
from dulwich.tests import TestCase
from ..hooks import CommitMsgShellHook, PostCommitShellHook, PreCommitShellHook
Tests for executing hooks.