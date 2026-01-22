import re
from unittest import mock
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class TestGenerateRandomString(common.HeatTestCase):
    scenarios = [('lettersdigits', dict(length=1, seq='lettersdigits', pattern='[a-zA-Z0-9]')), ('letters', dict(length=10, seq='letters', pattern='[a-zA-Z]')), ('lowercase', dict(length=100, seq='lowercase', pattern='[a-z]')), ('uppercase', dict(length=50, seq='uppercase', pattern='[A-Z]')), ('digits', dict(length=512, seq='digits', pattern='[0-9]')), ('hexdigits', dict(length=16, seq='hexdigits', pattern='[A-F0-9]')), ('octdigits', dict(length=32, seq='octdigits', pattern='[0-7]'))]
    template_rs = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  secret:\n    Type: OS::Heat::RandomString\n"

    def parse_stack(self, t):
        stack_name = 'test_stack'
        tmpl = template.Template(t)
        stack = parser.Stack(utils.dummy_context(), stack_name, tmpl)
        stack.validate()
        stack.store()
        return stack

    def test_generate_random_string_backward_compatible(self):
        stack = self.parse_stack(template_format.parse(self.template_rs))
        secret = stack['secret']
        char_classes = secret.properties['character_classes']
        for char_cl in char_classes:
            char_cl['class'] = self.seq
        for i in range(1, 32):
            r = secret._generate_random_string([], char_classes, self.length)
            self.assertThat(r, matchers.HasLength(self.length))
            regex = '%s{%s}' % (self.pattern, self.length)
            self.assertThat(r, matchers.MatchesRegex(regex))