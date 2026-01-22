from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
@skipUnless(MYPY_AVAILABLE, 'Tests require mypy to be installed.')
class MypyTests(TestCase):

    def test_mypy_working(self) -> None:
        """
        mypy's API is able to function and produce errors when expected.
        """
        _assert_mypy(True, 'ivar: int = 1\n')
        _assert_mypy(False, "ivar: int = 'bad'\n")

    def test_setup_no_args(self) -> None:
        """
        setup() and no_setup() take no arguments.
        """
        _assert_mypy(True, dedent('\n                from crochet import setup\n                setup()\n                '))
        _assert_mypy(True, dedent('\n                from crochet import no_setup\n                no_setup()\n                '))

    def test_run_in_reactor_func_takes_same_args(self) -> None:
        """
        The mypy plugin correctly passes the wrapped parameter signature through the
        @run_in_reactor decorator.
        """
        template = dedent('            from crochet import run_in_reactor\n\n            @run_in_reactor\n            def foo({params}) -> None:\n                pass\n\n            foo({args})\n            ')
        for params, args, good in (('x: int, y: str, z: float, *a: int, **kw: str', "1, 'something', -1, 4, 5, 6, k1='x', k2='y'", True), ('', '1', False), ('x: int', '', False), ('x: int', '1, 2', False), ('x: int', "'something'", False), ('*x: int', '1, 2, 3', True), ('*x: int', "'something'", False), ('**x: int', 'k1=16, k2=-5', True), ('**x: int', "k1='something'", False), ('x: int, y: str', "1, 'ok'", True), ('x: int, y: str', "'not ok', 1", False), ('x: str, y: int', "'ok', 1", True), ('x: str, y: int', "1, 'not ok'", False)):
            with self.subTest(params=params, args=args):
                _assert_mypy(good, template.format(params=params, args=args))

    def test_run_in_reactor_func_returns_typed_eventual(self) -> None:
        """
        run_in_reactor preserves the decorated function's return type indirectly
        through an EventualResult.
        """
        template = dedent('            from typing import Optional\n            from crochet import EventualResult, run_in_reactor\n\n            @run_in_reactor\n            def foo() -> {return_type}:\n                return {return_value}\n\n            eventual_result: {receiver_type} = foo()\n            final_result: {final_type} = eventual_result.wait(1)\n            ')
        for return_type, return_value, receiver_type, final_type, good in (('int', '1', 'EventualResult[int]', 'int', True), ('int', "'str'", 'EventualResult[int]', 'int', False), ('int', '1', 'EventualResult[str]', 'int', False), ('int', '1', 'EventualResult[str]', 'str', False), ('int', '1', 'int', 'int', False), ('int', '1', 'EventualResult[int]', 'Optional[int]', True), ('Optional[int]', '1', 'EventualResult[Optional[int]]', 'Optional[int]', True), ('Optional[int]', 'None', 'EventualResult[Optional[int]]', 'Optional[int]', True), ('Optional[int]', '1', 'EventualResult[int]', 'Optional[int]', False), ('Optional[int]', '1', 'EventualResult[Optional[int]]', 'int', False)):
            with self.subTest(return_type=return_type, return_value=return_value, receiver_type=receiver_type, final_type=final_type):
                _assert_mypy(good, template.format(return_type=return_type, return_value=return_value, receiver_type=receiver_type, final_type=final_type))

    def test_run_in_reactor_func_signature_transform(self) -> None:
        """
        The mypy plugin correctly passes the wrapped signature though the
        @run_in_reactor decorator with an EventualResult-wrapped return type.
        """
        template = dedent('            from typing import Callable\n            from crochet import EventualResult, run_in_reactor\n\n            class Thing:\n                pass\n\n            @run_in_reactor\n            def foo(x: int, y: str, z: float) -> Thing:\n                return Thing()\n\n            re_foo: {result_type} = foo\n            ')
        for result_type, good in (('Callable[[int, str, float], EventualResult[Thing]]', True), ('Callable[[int, str, float], EventualResult[object]]', True), ('Callable[[int, str, float], EventualResult[int]]', False), ('Callable[[int, str, float], Thing]', False), ('Callable[[int, str, float], int]', False), ('Callable[[int, str], EventualResult[Thing]]', False), ('Callable[[int], EventualResult[Thing]]', False), ('Callable[[], EventualResult[Thing]]', False), ('Callable[[float, int, str], EventualResult[Thing]]', False)):
            with self.subTest(result_type=result_type):
                _assert_mypy(good, template.format(result_type=result_type))

    def test_eventual_result_cancel_signature(self) -> None:
        """
        EventualResult's cancel() method takes no arguments.
        """
        _assert_mypy(True, dedent('                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> None:\n                    er.cancel()\n                '))

    def test_eventual_result_wait_signature(self) -> None:
        """
        EventualResult's wait() method takes one timeout float argument.
        """
        _assert_mypy(True, dedent('                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> object:\n                    return er.wait(2.0)\n                '))
        _assert_mypy(True, dedent('                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> object:\n                    return er.wait(timeout=2.0)\n                '))

    def test_eventual_result_stash_signature(self) -> None:
        """
        EventualResult's stash() method takes no arguments and returns the same type
        retrieve_result's one result_id parameter takes.
        """
        _assert_mypy(True, dedent('                from crochet import EventualResult, retrieve_result\n                def foo(er: EventualResult[object]) -> None:\n                    retrieve_result(er.stash())\n                    retrieve_result(result_id=er.stash())\n                '))

    def test_eventual_result_original_failure_signature(self) -> None:
        """
        EventualResult's original_failure() method takes no arguments and returns an
        optional Failure.
        """
        _assert_mypy(True, dedent('                from typing import Optional\n                from twisted.python.failure import Failure\n                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> Optional[Failure]:\n                    return er.original_failure()\n                '))
        _assert_mypy(False, dedent('                from twisted.python.failure import Failure\n                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> Failure:\n                    return er.original_failure()\n                '))

    def test_exceptions(self) -> None:
        """
        ReactorStopped and TimeoutError are Exception types.
        """
        _assert_mypy(True, dedent('                from crochet import ReactorStopped, TimeoutError\n                e1: Exception = ReactorStopped()\n                e2: Exception = TimeoutError()\n                '))

    def test_retrieve_result_returns_untyped_eventual_result(self) -> None:
        """
        retrieve_result() returns an untyped EventualResult.
        """
        _assert_mypy(True, dedent('                from crochet import EventualResult, retrieve_result\n                r: EventualResult[object] = retrieve_result(3)\n                '))
        _assert_mypy(False, dedent('                from crochet import EventualResult, retrieve_result\n                r: EventualResult[int] = retrieve_result(3)\n                '))

    def test_wait_for_signature(self) -> None:
        """
        The @wait_for decorator takes a timeout float.
        """
        _assert_mypy(True, dedent('                from crochet import wait_for\n\n                @wait_for(1.5)\n                def foo() -> None:\n                    pass\n                '))
        _assert_mypy(True, dedent('                from crochet import wait_for\n\n                @wait_for(timeout=1.5)\n                def foo() -> None:\n                    pass\n                '))

    def test_wait_for_func_signature_unchanged(self) -> None:
        """
        The @wait_for(timeout) decorator preserves the wrapped function's signature.
        """
        template = dedent('            from typing import Callable\n            from crochet import wait_for\n\n            class Thing:\n                pass\n\n            @wait_for(1)\n            def foo(x: int, y: str, z: float) -> Thing:\n                return Thing()\n\n            re_foo: {result_type} = foo\n            ')
        for result_type, good in (('Callable[[int, str, float], Thing]', True), ('Callable[[int, str, float], object]', True), ('Callable[[int, str, float], int]', False), ('Callable[[int, str, float], EventualResult[Thing]]', False), ('Callable[[int, str, float], None]', False), ('Callable[[int, str], Thing]', False), ('Callable[[int], Thing]', False), ('Callable[[], Thing]', False), ('Callable[[float, int, str], Thing]', False)):
            with self.subTest(result_type=result_type):
                _assert_mypy(good, template.format(result_type=result_type))

    def test_version_string(self) -> None:
        """
        __version__ is a string.
        """
        _assert_mypy(True, dedent('                import crochet\n                x: str = crochet.__version__\n                '))