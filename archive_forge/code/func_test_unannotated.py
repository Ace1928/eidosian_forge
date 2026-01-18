import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
def test_unannotated(self):
    c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
    self.repo.refs[b'HEAD'] = c3.id
    porcelain.tag_create(self.repo.path, b'tryme', annotated=False)
    tags = self.repo.refs.as_dict(b'refs/tags')
    self.assertEqual(list(tags.keys()), [b'tryme'])
    self.repo[b'refs/tags/tryme']
    self.assertEqual(list(tags.values()), [self.repo.head()])