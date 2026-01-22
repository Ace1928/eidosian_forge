from __future__ import (absolute_import, division, print_function)
import fnmatch
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
from ansible.module_utils.facts import collector
from ansible.module_utils.common.collections import is_string
class CollectorMetaDataCollector(collector.BaseFactCollector):
    """Collector that provides a facts with the gather_subset metadata."""
    name = 'gather_subset'
    _fact_ids = set()

    def __init__(self, collectors=None, namespace=None, gather_subset=None, module_setup=None):
        super(CollectorMetaDataCollector, self).__init__(collectors, namespace)
        self.gather_subset = gather_subset
        self.module_setup = module_setup

    def collect(self, module=None, collected_facts=None):
        meta_facts = {'gather_subset': self.gather_subset}
        if self.module_setup:
            meta_facts['module_setup'] = self.module_setup
        return meta_facts