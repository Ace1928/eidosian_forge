from novaclient import base
class AgentsManager(base.ManagerWithFind):
    resource_class = Agent

    def list(self, hypervisor=None):
        """List all agent builds."""
        url = '/os-agents'
        if hypervisor:
            url = '/os-agents?hypervisor=%s' % hypervisor
        return self._list(url, 'agents')

    def _build_update_body(self, version, url, md5hash):
        return {'para': {'version': version, 'url': url, 'md5hash': md5hash}}

    def update(self, id, version, url, md5hash):
        """Update an existing agent build."""
        body = self._build_update_body(version, url, md5hash)
        return self._update('/os-agents/%s' % id, body, 'agent')

    def create(self, os, architecture, version, url, md5hash, hypervisor):
        """Create a new agent build."""
        body = {'agent': {'hypervisor': hypervisor, 'os': os, 'architecture': architecture, 'version': version, 'url': url, 'md5hash': md5hash}}
        return self._create('/os-agents', body, 'agent')

    def delete(self, id):
        """
        Deletes an existing agent build.

        :param id: The agent's id to delete
        :returns: An instance of novaclient.base.TupleWithMeta
        """
        return self._delete('/os-agents/%s' % id)