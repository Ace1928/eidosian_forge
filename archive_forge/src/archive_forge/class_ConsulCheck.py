from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
class ConsulCheck(object):

    def __init__(self, check_id, name, node=None, host='localhost', script=None, interval=None, ttl=None, notes=None, tcp=None, http=None, timeout=None, service_id=None):
        self.check_id = self.name = name
        if check_id:
            self.check_id = check_id
        self.service_id = service_id
        self.notes = notes
        self.node = node
        self.host = host
        self.interval = self.validate_duration('interval', interval)
        self.ttl = self.validate_duration('ttl', ttl)
        self.script = script
        self.tcp = tcp
        self.http = http
        self.timeout = self.validate_duration('timeout', timeout)
        self.check = None
        if script:
            self.check = consul.Check.script(script, self.interval)
        if ttl:
            self.check = consul.Check.ttl(self.ttl)
        if http:
            self.check = consul.Check.http(http, self.interval, self.timeout)
        if tcp:
            regex = '(?P<host>.*):(?P<port>(?:[0-9]+))$'
            match = re.match(regex, tcp)
            if not match:
                raise Exception('tcp check must be in host:port format')
            self.check = consul.Check.tcp(match.group('host').strip('[]'), int(match.group('port')), self.interval)

    def validate_duration(self, name, duration):
        if duration:
            duration_units = ['ns', 'us', 'ms', 's', 'm', 'h']
            if not any((duration.endswith(suffix) for suffix in duration_units)):
                duration = '{0}s'.format(duration)
        return duration

    def register(self, consul_api):
        consul_api.agent.check.register(self.name, check_id=self.check_id, service_id=self.service_id, notes=self.notes, check=self.check)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.check_id == other.check_id and (self.service_id == other.service_id) and (self.name == other.name) and (self.script == other.script) and (self.interval == other.interval)

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self):
        data = {}
        self._add(data, 'id', attr='check_id')
        self._add(data, 'name', attr='check_name')
        self._add(data, 'script')
        self._add(data, 'node')
        self._add(data, 'notes')
        self._add(data, 'host')
        self._add(data, 'interval')
        self._add(data, 'ttl')
        self._add(data, 'tcp')
        self._add(data, 'http')
        self._add(data, 'timeout')
        self._add(data, 'service_id')
        return data

    def _add(self, data, key, attr=None):
        try:
            if attr is None:
                attr = key
            data[key] = getattr(self, attr)
        except Exception:
            pass