import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
class TestTryImport(TestCase):

    def test_doesnt_exist(self):
        marker = object()
        result = try_import('doesntexist', marker)
        self.assertThat(result, Is(marker))

    def test_None_is_default_alternative(self):
        result = try_import('doesntexist')
        self.assertThat(result, Is(None))

    def test_existing_module(self):
        result = try_import('os', object())
        import os
        self.assertThat(result, Is(os))

    def test_existing_submodule(self):
        result = try_import('os.path', object())
        import os
        self.assertThat(result, Is(os.path))

    def test_nonexistent_submodule(self):
        marker = object()
        result = try_import('os.doesntexist', marker)
        self.assertThat(result, Is(marker))

    def test_object_from_module(self):
        result = try_import('os.path.join')
        import os
        self.assertThat(result, Is(os.path.join))

    def test_error_callback(self):
        check_error_callback(self, try_import, 'doesntexist', 1, False)

    def test_error_callback_missing_module_member(self):
        check_error_callback(self, try_import, 'os.nonexistent', 1, False)

    def test_error_callback_not_on_success(self):
        check_error_callback(self, try_import, 'os.path', 0, True)

    def test_handle_partly_imported_name(self):
        outer = types.ModuleType('extras.outer')
        inner = types.ModuleType('extras.outer.inner')
        inner.attribute = object()
        self.addCleanup(sys.modules.pop, 'extras.outer', None)
        self.addCleanup(sys.modules.pop, 'extras.outer.inner', None)
        sys.modules['extras.outer'] = outer
        sys.modules['extras.outer.inner'] = inner
        result = try_import('extras.outer.inner.attribute')
        self.expectThat(result, Is(inner.attribute))
        result = try_import('extras.outer.inner')
        self.expectThat(result, Is(inner))