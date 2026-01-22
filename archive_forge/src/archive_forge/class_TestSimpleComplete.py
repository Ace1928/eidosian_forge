import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
class TestSimpleComplete(unittest.TestCase):

    def setUp(self):
        self.module_gatherer = ModuleGatherer()
        self.module_gatherer.modules = ['zzabc', 'zzabd', 'zzefg', 'zzabc.e', 'zzabc.f', 'zzefg.a1', 'zzefg.a2']

    def test_simple_completion(self):
        self.assertSetEqual(self.module_gatherer.complete(10, 'import zza'), {'zzabc', 'zzabd'})
        self.assertSetEqual(self.module_gatherer.complete(11, 'import  zza'), {'zzabc', 'zzabd'})

    def test_import_empty(self):
        self.assertSetEqual(self.module_gatherer.complete(13, 'import zzabc.'), {'zzabc.e', 'zzabc.f'})
        self.assertSetEqual(self.module_gatherer.complete(14, 'import  zzabc.'), {'zzabc.e', 'zzabc.f'})

    def test_import(self):
        self.assertSetEqual(self.module_gatherer.complete(14, 'import zzefg.a'), {'zzefg.a1', 'zzefg.a2'})
        self.assertSetEqual(self.module_gatherer.complete(15, 'import  zzefg.a'), {'zzefg.a1', 'zzefg.a2'})

    @unittest.expectedFailure
    def test_import_blank(self):
        self.assertSetEqual(self.module_gatherer.complete(7, 'import '), {'zzabc', 'zzabd', 'zzefg'})
        self.assertSetEqual(self.module_gatherer.complete(8, 'import  '), {'zzabc', 'zzabd', 'zzefg'})

    @unittest.expectedFailure
    def test_from_import_empty(self):
        self.assertSetEqual(self.module_gatherer.complete(5, 'from '), {'zzabc', 'zzabd', 'zzefg'})
        self.assertSetEqual(self.module_gatherer.complete(6, 'from  '), {'zzabc', 'zzabd', 'zzefg'})

    @unittest.expectedFailure
    def test_from_module_import_empty(self):
        self.assertSetEqual(self.module_gatherer.complete(18, 'from zzabc import '), {'e', 'f'})
        self.assertSetEqual(self.module_gatherer.complete(19, 'from  zzabc import '), {'e', 'f'})
        self.assertSetEqual(self.module_gatherer.complete(19, 'from zzabc  import '), {'e', 'f'})
        self.assertSetEqual(self.module_gatherer.complete(19, 'from zzabc import  '), {'e', 'f'})

    def test_from_module_import(self):
        self.assertSetEqual(self.module_gatherer.complete(19, 'from zzefg import a'), {'a1', 'a2'})
        self.assertSetEqual(self.module_gatherer.complete(20, 'from  zzefg import a'), {'a1', 'a2'})
        self.assertSetEqual(self.module_gatherer.complete(20, 'from zzefg  import a'), {'a1', 'a2'})
        self.assertSetEqual(self.module_gatherer.complete(20, 'from zzefg import  a'), {'a1', 'a2'})