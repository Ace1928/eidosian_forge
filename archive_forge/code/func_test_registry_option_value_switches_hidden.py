import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_registry_option_value_switches_hidden(self):
    reg = registry.Registry()

    class Hider:
        hidden = True
    reg.register('new', 1, 'Current.')
    reg.register('old', 0, 'Legacy.', info=Hider())
    opt = option.RegistryOption('protocol', 'Talking.', reg, value_switches=True, enum_switch=False)
    pot = self.pot_from_option(opt)
    self.assertContainsString(pot, '\n# help of \'protocol\' test\nmsgid "Talking."\n')
    self.assertContainsString(pot, '\n# help of \'protocol=new\' test\nmsgid "Current."\n')
    self.assertNotContainsString(pot, "'protocol=old'")