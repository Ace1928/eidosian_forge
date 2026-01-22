import os
from ansible.plugins.lookup import LookupBase
import ansible_collections.cloud.common.plugins.module_utils.turbo.common
from ansible_collections.cloud.common.plugins.module_utils.turbo.exceptions import (
class TurboLookupBase(LookupBase):

    def run_on_daemon(self, terms, variables=None, **kwargs):
        self._ttl = get_server_ttl(variables)
        return self.execute(terms=terms, variables=variables, **kwargs)

    @property
    def socket_path(self):
        if not hasattr(self, '__socket_path'):
            '\n            Input:\n                _load_name: ansible_collections.cloud.common.plugins.lookup.turbo_random_lookup\n            Output:\n                __socket_path: {HOME}/.ansible/tmp/turbo_mode_cloud.common.socket\n            this will allow to have one socket per collection\n            '
            name = self._load_name
            ansible_collections = 'ansible_collections.'
            if name.startswith(ansible_collections):
                name = name.replace(ansible_collections, '', 1)
                lookup_plugins = '.plugins.lookup.'
                idx = name.find(lookup_plugins)
                if idx != -1:
                    name = name[:idx]
            self.__socket_path = os.environ['HOME'] + '/.ansible/tmp/turbo_lookup.{0}.socket'.format(name)
        return self.__socket_path

    def execute(self, terms, variables=None, **kwargs):
        result = None
        with ansible_collections.cloud.common.plugins.module_utils.turbo.common.connect(socket_path=self.socket_path, ttl=self._ttl, plugin='lookup') as turbo_socket:
            content = (self._load_name, terms, variables, kwargs)
            result, errors = turbo_socket.communicate(content)
            if errors:
                raise EmbeddedModuleUnexpectedFailure(errors)
            return result