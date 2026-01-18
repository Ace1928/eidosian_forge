from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_wait_for_signature(self) -> None:
    """
        The @wait_for decorator takes a timeout float.
        """
    _assert_mypy(True, dedent('                from crochet import wait_for\n\n                @wait_for(1.5)\n                def foo() -> None:\n                    pass\n                '))
    _assert_mypy(True, dedent('                from crochet import wait_for\n\n                @wait_for(timeout=1.5)\n                def foo() -> None:\n                    pass\n                '))