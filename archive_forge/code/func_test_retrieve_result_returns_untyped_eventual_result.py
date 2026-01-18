from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_retrieve_result_returns_untyped_eventual_result(self) -> None:
    """
        retrieve_result() returns an untyped EventualResult.
        """
    _assert_mypy(True, dedent('                from crochet import EventualResult, retrieve_result\n                r: EventualResult[object] = retrieve_result(3)\n                '))
    _assert_mypy(False, dedent('                from crochet import EventualResult, retrieve_result\n                r: EventualResult[int] = retrieve_result(3)\n                '))