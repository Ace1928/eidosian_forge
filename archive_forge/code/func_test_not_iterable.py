import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_not_iterable(self):
    """
        Verify that assignment to nested tuples works correctly.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:for each="item in foo">\n            $item\n          </py:for>\n        </doc>', filename='test.html')
    try:
        list(tmpl.generate(foo=12))
        self.fail('Expected TemplateRuntimeError')
    except TypeError as e:
        assert str(e) == 'iteration over non-sequence' or str(e) == "'int' object is not iterable"
        exc_type, exc_value, exc_traceback = sys.exc_info()
        frame = exc_traceback.tb_next
        frames = []
        while frame.tb_next:
            frame = frame.tb_next
            frames.append(frame)
        expected_iter_str = "u'iter(foo)'" if IS_PYTHON2 else "'iter(foo)'"
        self.assertEqual('<Expression %s>' % expected_iter_str, frames[-1].tb_frame.f_code.co_name)
        self.assertEqual('test.html', frames[-1].tb_frame.f_code.co_filename)
        self.assertEqual(2, frames[-1].tb_lineno)