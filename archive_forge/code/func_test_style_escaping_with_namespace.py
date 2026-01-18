import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_style_escaping_with_namespace(self):
    text = '<style xmlns="http://www.w3.org/1999/xhtml">\n            html &gt; body { display: none; }\n        </style>'
    output = XML(text).render(HTMLSerializer, encoding=None)
    self.assertEqual('<style>\n            html > body { display: none; }\n        </style>', output)