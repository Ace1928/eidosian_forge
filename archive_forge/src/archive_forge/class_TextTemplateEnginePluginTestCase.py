import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
class TextTemplateEnginePluginTestCase(unittest.TestCase):

    def test_init_no_options(self):
        plugin = TextTemplateEnginePlugin()
        self.assertEqual(None, plugin.default_encoding)
        self.assertEqual('text', plugin.default_format)
        self.assertEqual([], plugin.loader.search_path)
        self.assertEqual(True, plugin.loader.auto_reload)
        self.assertEqual(25, plugin.loader._cache.capacity)

    def test_init_with_loader_options(self):
        plugin = TextTemplateEnginePlugin(options={'genshi.auto_reload': 'off', 'genshi.max_cache_size': '100', 'genshi.search_path': '/usr/share/tmpl:/usr/local/share/tmpl'})
        self.assertEqual(['/usr/share/tmpl', '/usr/local/share/tmpl'], plugin.loader.search_path)
        self.assertEqual(False, plugin.loader.auto_reload)
        self.assertEqual(100, plugin.loader._cache.capacity)

    def test_init_with_output_options(self):
        plugin = TextTemplateEnginePlugin(options={'genshi.default_encoding': 'iso-8859-15'})
        self.assertEqual('iso-8859-15', plugin.default_encoding)

    def test_init_with_new_syntax(self):
        plugin = TextTemplateEnginePlugin(options={'genshi.new_text_syntax': 'yes'})
        self.assertEqual(NewTextTemplate, plugin.template_class)
        tmpl = plugin.load_template(PACKAGE + '.templates.new_syntax')
        output = plugin.render({'foo': True}, template=tmpl)
        self.assertEqual('bar', output)

    def test_load_template_from_file(self):
        plugin = TextTemplateEnginePlugin()
        tmpl = plugin.load_template(PACKAGE + '.templates.test')
        assert isinstance(tmpl, TextTemplate)
        self.assertEqual('test.txt', os.path.basename(tmpl.filename))

    def test_load_template_from_string(self):
        plugin = TextTemplateEnginePlugin()
        tmpl = plugin.load_template(None, template_string='$message')
        self.assertEqual(None, tmpl.filename)
        assert isinstance(tmpl, TextTemplate)

    def test_transform_without_load(self):
        plugin = TextTemplateEnginePlugin()
        stream = plugin.transform({'message': 'Hello'}, PACKAGE + '.templates.test')
        assert isinstance(stream, Stream)

    def test_transform_with_load(self):
        plugin = TextTemplateEnginePlugin()
        tmpl = plugin.load_template(PACKAGE + '.templates.test')
        stream = plugin.transform({'message': 'Hello'}, tmpl)
        assert isinstance(stream, Stream)

    def test_render(self):
        plugin = TextTemplateEnginePlugin()
        tmpl = plugin.load_template(PACKAGE + '.templates.test')
        output = plugin.render({'message': 'Hello'}, template=tmpl)
        self.assertEqual('Test\n====\n\nHello\n', output)

    def test_helper_functions(self):
        plugin = TextTemplateEnginePlugin()
        tmpl = plugin.load_template(PACKAGE + '.templates.functions')
        output = plugin.render({}, template=tmpl)
        self.assertEqual('False\nbar\n', output)