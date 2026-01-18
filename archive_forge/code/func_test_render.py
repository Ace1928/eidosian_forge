import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_render(self):
    plugin = TextTemplateEnginePlugin()
    tmpl = plugin.load_template(PACKAGE + '.templates.test')
    output = plugin.render({'message': 'Hello'}, template=tmpl)
    self.assertEqual('Test\n====\n\nHello\n', output)