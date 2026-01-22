import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
class FindShallowTests(TestCase):

    def setUp(self):
        super().setUp()
        self._store = MemoryObjectStore()

    def make_commit(self, **attrs):
        commit = make_commit(**attrs)
        self._store.add_object(commit)
        return commit

    def make_linear_commits(self, n, message=b''):
        commits = []
        parents = []
        for _ in range(n):
            commits.append(self.make_commit(parents=parents, message=message))
            parents = [commits[-1].id]
        return commits

    def assertSameElements(self, expected, actual):
        self.assertEqual(set(expected), set(actual))

    def test_linear(self):
        c1, c2, c3 = self.make_linear_commits(3)
        self.assertEqual(({c3.id}, set()), _find_shallow(self._store, [c3.id], 1))
        self.assertEqual(({c2.id}, {c3.id}), _find_shallow(self._store, [c3.id], 2))
        self.assertEqual(({c1.id}, {c2.id, c3.id}), _find_shallow(self._store, [c3.id], 3))
        self.assertEqual((set(), {c1.id, c2.id, c3.id}), _find_shallow(self._store, [c3.id], 4))

    def test_multiple_independent(self):
        a = self.make_linear_commits(2, message=b'a')
        b = self.make_linear_commits(2, message=b'b')
        c = self.make_linear_commits(2, message=b'c')
        heads = [a[1].id, b[1].id, c[1].id]
        self.assertEqual(({a[0].id, b[0].id, c[0].id}, set(heads)), _find_shallow(self._store, heads, 2))

    def test_multiple_overlapping(self):
        c1, c2 = self.make_linear_commits(2)
        c3 = self.make_commit(parents=[c1.id])
        c4 = self.make_commit(parents=[c3.id])
        self.assertEqual(({c1.id}, {c1.id, c2.id, c3.id, c4.id}), _find_shallow(self._store, [c2.id, c4.id], 3))

    def test_merge(self):
        c1 = self.make_commit()
        c2 = self.make_commit()
        c3 = self.make_commit(parents=[c1.id, c2.id])
        self.assertEqual(({c1.id, c2.id}, {c3.id}), _find_shallow(self._store, [c3.id], 2))

    def test_tag(self):
        c1, c2 = self.make_linear_commits(2)
        tag = make_tag(c2, name=b'tag')
        self._store.add_object(tag)
        self.assertEqual(({c1.id}, {c2.id}), _find_shallow(self._store, [tag.id], 2))