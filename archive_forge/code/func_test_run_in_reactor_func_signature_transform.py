from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_run_in_reactor_func_signature_transform(self) -> None:
    """
        The mypy plugin correctly passes the wrapped signature though the
        @run_in_reactor decorator with an EventualResult-wrapped return type.
        """
    template = dedent('            from typing import Callable\n            from crochet import EventualResult, run_in_reactor\n\n            class Thing:\n                pass\n\n            @run_in_reactor\n            def foo(x: int, y: str, z: float) -> Thing:\n                return Thing()\n\n            re_foo: {result_type} = foo\n            ')
    for result_type, good in (('Callable[[int, str, float], EventualResult[Thing]]', True), ('Callable[[int, str, float], EventualResult[object]]', True), ('Callable[[int, str, float], EventualResult[int]]', False), ('Callable[[int, str, float], Thing]', False), ('Callable[[int, str, float], int]', False), ('Callable[[int, str], EventualResult[Thing]]', False), ('Callable[[int], EventualResult[Thing]]', False), ('Callable[[], EventualResult[Thing]]', False), ('Callable[[float, int, str], EventualResult[Thing]]', False)):
        with self.subTest(result_type=result_type):
            _assert_mypy(good, template.format(result_type=result_type))