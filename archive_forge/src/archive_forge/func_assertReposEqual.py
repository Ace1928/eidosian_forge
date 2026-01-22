import errno
import functools
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Tuple
from dulwich.tests import SkipTest, TestCase
from ...protocol import TCP_GIT_PORT
from ...repo import Repo
def assertReposEqual(self, repo1, repo2):
    self.assertEqual(repo1.get_refs(), repo2.get_refs())
    self.assertObjectStoreEqual(repo1.object_store, repo2.object_store)