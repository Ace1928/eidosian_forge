from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
class ConsulService(object):

    def __init__(self, service_id=None, name=None, address=None, port=-1, tags=None, loaded=None):
        self.id = self.name = name
        if service_id:
            self.id = service_id
        self.address = address
        self.port = port
        self.tags = tags
        self._checks = []
        if loaded:
            self.id = loaded['ID']
            self.name = loaded['Service']
            self.port = loaded['Port']
            self.tags = loaded['Tags']

    def register(self, consul_api):
        optional = {}
        if self.port:
            optional['port'] = self.port
        if len(self._checks) > 0:
            optional['check'] = self._checks[0].check
        consul_api.agent.service.register(self.name, service_id=self.id, address=self.address, tags=self.tags, **optional)

    def add_check(self, check):
        self._checks.append(check)

    def checks(self):
        return self._checks

    def has_checks(self):
        return len(self._checks) > 0

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.id == other.id and (self.name == other.name) and (self.port == other.port) and (self.tags == other.tags)

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self):
        data = {'id': self.id, 'name': self.name}
        if self.port:
            data['port'] = self.port
        if self.tags and len(self.tags) > 0:
            data['tags'] = self.tags
        if len(self._checks) > 0:
            data['check'] = self._checks[0].to_dict()
        return data