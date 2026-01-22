from __future__ import (absolute_import, division, print_function)
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
class PythonFactCollector(BaseFactCollector):
    name = 'python'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        python_facts = {}
        python_facts['python'] = {'version': {'major': sys.version_info[0], 'minor': sys.version_info[1], 'micro': sys.version_info[2], 'releaselevel': sys.version_info[3], 'serial': sys.version_info[4]}, 'version_info': list(sys.version_info), 'executable': sys.executable, 'has_sslcontext': HAS_SSLCONTEXT}
        try:
            python_facts['python']['type'] = sys.subversion[0]
        except AttributeError:
            try:
                python_facts['python']['type'] = sys.implementation.name
            except AttributeError:
                python_facts['python']['type'] = None
        return python_facts