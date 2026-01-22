import io
import tempfile
import textwrap
import six
from six.moves import configparser
import sys
from pbr.tests import base
from pbr import util
class TestExtrasRequireParsingScenarios(base.BaseTestCase):
    scenarios = [('simple_extras', {'config_text': '\n                [extras]\n                first =\n                    foo\n                    bar==1.0\n                second =\n                    baz>=3.2\n                    foo\n                ', 'expected_extra_requires': {'first': ['foo', 'bar==1.0'], 'second': ['baz>=3.2', 'foo'], 'test': ['requests-mock'], "test:(python_version=='2.6')": ['ordereddict']}}), ('with_markers', {'config_text': "\n                [extras]\n                test =\n                    foo:python_version=='2.6'\n                    bar\n                    baz<1.6 :python_version=='2.6'\n                    zaz :python_version>'1.0'\n                ", 'expected_extra_requires': {"test:(python_version=='2.6')": ['foo', 'baz<1.6'], 'test': ['bar', 'zaz']}}), ('no_extras', {'config_text': '\n            [metadata]\n            long_description = foo\n            ', 'expected_extra_requires': {}})]

    def test_extras_parsing(self):
        config = config_from_ini(self.config_text)
        kwargs = util.setup_cfg_to_setup_kwargs(config)
        self.assertEqual(self.expected_extra_requires, kwargs['extras_require'])