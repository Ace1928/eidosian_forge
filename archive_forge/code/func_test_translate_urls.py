import copy
from testtools import matchers
from keystone.common import json_home
from keystone.tests import unit
def test_translate_urls(self):
    href_rel = self.getUniqueString()
    href = self.getUniqueString()
    href_template_rel = self.getUniqueString()
    href_template = self.getUniqueString()
    href_vars = {self.getUniqueString(): self.getUniqueString()}
    original_json_home = {'resources': {href_rel: {'href': href}, href_template_rel: {'href-template': href_template, 'href-vars': href_vars}}}
    new_json_home = copy.deepcopy(original_json_home)
    new_prefix = self.getUniqueString()
    json_home.translate_urls(new_json_home, new_prefix)
    exp_json_home = {'resources': {href_rel: {'href': new_prefix + href}, href_template_rel: {'href-template': new_prefix + href_template, 'href-vars': href_vars}}}
    self.assertThat(new_json_home, matchers.Equals(exp_json_home))