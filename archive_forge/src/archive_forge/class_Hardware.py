from __future__ import (absolute_import, division, print_function)
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
class Hardware:
    platform = 'Generic'

    def __init__(self, module, load_on_init=False):
        self.module = module

    def populate(self, collected_facts=None):
        return {}