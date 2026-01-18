from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_used_package_with_submodule_import_of_alias(self):
    """
        Usage of package by alias marks submodule imports as used.
        """
    self.flakes('\n        import foo as f\n        import foo.bar\n        f.bar.do_something()\n        ')
    self.flakes('\n        import foo as f\n        import foo.bar.blah\n        f.bar.blah.do_something()\n        ')