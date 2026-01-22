from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
class TestImportationObject(TestCase):

    def test_import_basic(self):
        binding = Importation('a', None, 'a')
        assert binding.source_statement == 'import a'
        assert str(binding) == 'a'

    def test_import_as(self):
        binding = Importation('c', None, 'a')
        assert binding.source_statement == 'import a as c'
        assert str(binding) == 'a as c'

    def test_import_submodule(self):
        binding = SubmoduleImportation('a.b', None)
        assert binding.source_statement == 'import a.b'
        assert str(binding) == 'a.b'

    def test_import_submodule_as(self):
        binding = Importation('c', None, 'a.b')
        assert binding.source_statement == 'import a.b as c'
        assert str(binding) == 'a.b as c'

    def test_import_submodule_as_source_name(self):
        binding = Importation('a', None, 'a.b')
        assert binding.source_statement == 'import a.b as a'
        assert str(binding) == 'a.b as a'

    def test_importfrom_relative(self):
        binding = ImportationFrom('a', None, '.', 'a')
        assert binding.source_statement == 'from . import a'
        assert str(binding) == '.a'

    def test_importfrom_relative_parent(self):
        binding = ImportationFrom('a', None, '..', 'a')
        assert binding.source_statement == 'from .. import a'
        assert str(binding) == '..a'

    def test_importfrom_relative_with_module(self):
        binding = ImportationFrom('b', None, '..a', 'b')
        assert binding.source_statement == 'from ..a import b'
        assert str(binding) == '..a.b'

    def test_importfrom_relative_with_module_as(self):
        binding = ImportationFrom('c', None, '..a', 'b')
        assert binding.source_statement == 'from ..a import b as c'
        assert str(binding) == '..a.b as c'

    def test_importfrom_member(self):
        binding = ImportationFrom('b', None, 'a', 'b')
        assert binding.source_statement == 'from a import b'
        assert str(binding) == 'a.b'

    def test_importfrom_submodule_member(self):
        binding = ImportationFrom('c', None, 'a.b', 'c')
        assert binding.source_statement == 'from a.b import c'
        assert str(binding) == 'a.b.c'

    def test_importfrom_member_as(self):
        binding = ImportationFrom('c', None, 'a', 'b')
        assert binding.source_statement == 'from a import b as c'
        assert str(binding) == 'a.b as c'

    def test_importfrom_submodule_member_as(self):
        binding = ImportationFrom('d', None, 'a.b', 'c')
        assert binding.source_statement == 'from a.b import c as d'
        assert str(binding) == 'a.b.c as d'

    def test_importfrom_star(self):
        binding = StarImportation('a.b', None)
        assert binding.source_statement == 'from a.b import *'
        assert str(binding) == 'a.b.*'

    def test_importfrom_star_relative(self):
        binding = StarImportation('.b', None)
        assert binding.source_statement == 'from .b import *'
        assert str(binding) == '.b.*'

    def test_importfrom_future(self):
        binding = FutureImportation('print_function', None, None)
        assert binding.source_statement == 'from __future__ import print_function'
        assert str(binding) == '__future__.print_function'

    def test_unusedImport_underscore(self):
        """
        The magic underscore var should be reported as unused when used as an
        import alias.
        """
        self.flakes('import fu as _', m.UnusedImport)