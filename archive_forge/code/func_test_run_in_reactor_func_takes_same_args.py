from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_run_in_reactor_func_takes_same_args(self) -> None:
    """
        The mypy plugin correctly passes the wrapped parameter signature through the
        @run_in_reactor decorator.
        """
    template = dedent('            from crochet import run_in_reactor\n\n            @run_in_reactor\n            def foo({params}) -> None:\n                pass\n\n            foo({args})\n            ')
    for params, args, good in (('x: int, y: str, z: float, *a: int, **kw: str', "1, 'something', -1, 4, 5, 6, k1='x', k2='y'", True), ('', '1', False), ('x: int', '', False), ('x: int', '1, 2', False), ('x: int', "'something'", False), ('*x: int', '1, 2, 3', True), ('*x: int', "'something'", False), ('**x: int', 'k1=16, k2=-5', True), ('**x: int', "k1='something'", False), ('x: int, y: str', "1, 'ok'", True), ('x: int, y: str', "'not ok', 1", False), ('x: str, y: int', "'ok', 1", True), ('x: str, y: int', "1, 'not ok'", False)):
        with self.subTest(params=params, args=args):
            _assert_mypy(good, template.format(params=params, args=args))