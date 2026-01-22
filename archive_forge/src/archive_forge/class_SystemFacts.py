from __future__ import absolute_import, division, print_function
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.argspec.system.system import SystemArgs
class SystemFacts(object):
    """ The fortios system facts class
    """

    def __init__(self, module, fos=None, subset=None, subspec='config', options='options'):
        self._module = module
        self._fos = fos
        self._subset = subset

    def populate_facts(self, connection, ansible_facts, data=None):
        """ Populate the facts for system
        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :rtype: dictionary
        :returns: facts
        """
        ansible_facts['ansible_network_resources'].pop('system', None)
        facts = {}
        if self._subset['fact'].startswith(tuple(SystemArgs.FACT_SYSTEM_SUBSETS)):
            gather_method = getattr(self, self._subset['fact'].replace('-', '_'), self.system_fact)
            resp = gather_method()
            facts.update({self._subset['fact']: resp})
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts

    def system_fact(self):
        fos = self._fos
        vdom = self._module.params['vdom']
        return fos.monitor('system', self._subset['fact'][len('system_'):].replace('_', '/'), vdom=vdom)

    def system_interface_select(self):
        fos = self._fos
        vdom = self._module.params['vdom']
        query_string = '?vdom=' + vdom
        system_interface_select_param = self._subset['filters']
        if system_interface_select_param:
            for filter in system_interface_select_param:
                for key, val in filter.items():
                    if val:
                        query_string += '&' + str(key) + '=' + str(val)
        return fos.monitor('system', self._subset['fact'][len('system_'):].replace('_', '/') + query_string, vdom=None)