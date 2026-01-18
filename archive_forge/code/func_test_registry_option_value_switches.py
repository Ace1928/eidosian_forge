import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_registry_option_value_switches(self):
    opt = option.RegistryOption.from_kwargs('switch', help='Flip one.', value_switches=True, enum_switch=False, red='Big.', green='Small.')
    pot = self.pot_from_option(opt)
    self.assertContainsString(pot, '\n# help of \'switch\' test\nmsgid "Flip one."\n')
    self.assertContainsString(pot, '\n# help of \'switch=red\' test\nmsgid "Big."\n')
    self.assertContainsString(pot, '\n# help of \'switch=green\' test\nmsgid "Small."\n')