import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_transform_with_load(self):
    plugin = TextTemplateEnginePlugin()
    tmpl = plugin.load_template(PACKAGE + '.templates.test')
    stream = plugin.transform({'message': 'Hello'}, tmpl)
    assert isinstance(stream, Stream)