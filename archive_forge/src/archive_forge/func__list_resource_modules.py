from __future__ import absolute_import, division, print_function
import copy
import glob
import os
from importlib import import_module
from ansible.errors import AnsibleActionFail, AnsibleError
from ansible.module_utils._text import to_text
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.action.network import (
def _list_resource_modules(self):
    result = {}
    resource_modules = []
    self._cref = dict(zip(['corg', 'cname', 'plugin'], self._os_name.split('.')))
    fact_modulelib = 'ansible_collections.{corg}.{cname}.plugins.module_utils.network.{plugin}.facts.facts'.format(corg=self._cref['corg'], cname=self._cref['cname'], plugin=self._cref['plugin'])
    try:
        display.vvvv('fetching facts list from path %s' % fact_modulelib)
        facts_resource_subset = getattr(import_module(fact_modulelib), 'FACT_RESOURCE_SUBSETS')
        resource_modules = sorted(facts_resource_subset.keys())
    except ModuleNotFoundError:
        display.vvvv("'%s' is not defined" % fact_modulelib)
    except AttributeError:
        display.vvvv("'FACT_RESOURCE_SUBSETS is not defined in '%s'" % fact_modulelib)
    if not resource_modules:
        modulelib = 'ansible_collections.{corg}.{cname}.plugins.modules'.format(corg=self._cref['corg'], cname=self._cref['cname'])
        module_dir_path = os.path.dirname(import_module(modulelib).__file__)
        module_paths = glob.glob('{module_dir_path}/[!_]*.py'.format(module_dir_path=module_dir_path))
        for module_path in module_paths:
            module_name = os.path.basename(module_path).split('.')[0]
            docs = None
            try:
                display.vvvv("reading 'DOCUMENTATION' from path %s" % module_path)
                docs = getattr(import_module('%s.%s' % (modulelib, module_name)), 'DOCUMENTATION')
            except ModuleNotFoundError:
                display.vvvv("'%s' is not defined" % fact_modulelib)
            except AttributeError:
                display.vvvv("'DOCUMENTATION is not defined in '%s'" % fact_modulelib)
            if docs:
                if self._is_resource_module(docs):
                    resource_modules.append(module_name.split('_', 1)[1])
                else:
                    display.vvvvv("module in path '%s' is not a resource module" % module_path)
    result.update({'modules': sorted(resource_modules)})
    return result