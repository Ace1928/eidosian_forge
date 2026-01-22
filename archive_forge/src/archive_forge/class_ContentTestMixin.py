import doctest
import unittest
import six
from genshi import HTML
from genshi.builder import Element
from genshi.compat import IS_PYTHON2
from genshi.core import START, END, TEXT, QName, Attrs
from genshi.filters.transform import Transformer, StreamBuffer, ENTER, EXIT, \
import genshi.filters.transform
from genshi.tests.test_utils import doctest_suite
class ContentTestMixin(object):

    def _apply(self, select, content=None, html=FOOBAR):

        class Injector(object):
            count = 0

            def __iter__(self):
                self.count += 1
                return iter(HTML(u'CONTENT %i' % self.count))
        is_py2_str = isinstance(html, str) and IS_PYTHON2
        if is_py2_str:
            html = HTML(html, encoding='utf-8')
        else:
            html = HTML(html)
        if content is None:
            content = Injector()
        elif isinstance(content, six.string_types):
            content = HTML(content)
        return _transform(html, getattr(Transformer(select), self.operation)(content))