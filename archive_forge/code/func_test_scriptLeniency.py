from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_scriptLeniency(self) -> None:
    s = '\n        <script>(foo < bar) and (bar > foo)</script>\n        <script language="javascript">foo </scrip bar </script>\n        <script src="foo">\n        <script src="foo">baz</script>\n        <script /><script></script>\n        '
    d = microdom.parseString(s, beExtremelyLenient=1)
    self.assertEqual(d.firstChild().firstChild().firstChild().data, '(foo < bar) and (bar > foo)')
    self.assertEqual(d.firstChild().getElementsByTagName('script')[1].firstChild().data, 'foo </scrip bar ')