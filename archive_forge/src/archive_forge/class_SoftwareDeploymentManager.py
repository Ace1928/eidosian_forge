from urllib import parse
from heatclient.common import base
from heatclient.common import utils
class SoftwareDeploymentManager(base.BaseManager):
    resource_class = SoftwareDeployment

    def list(self, **kwargs):
        """Get a list of software deployments.

        :rtype: list of :class:`SoftwareDeployment`
        """
        url = '/software_deployments?%s' % parse.urlencode(kwargs)
        return self._list(url, 'software_deployments')

    def metadata(self, server_id):
        """Get a collection of software deployment metadata for given server.

        :rtype: list of :class:`SoftwareDeployment`
        """
        url = '/software_deployments/metadata/%s' % parse.quote(server_id)
        resp = self.client.get(url)
        body = utils.get_response_body(resp)
        return body.get('metadata')

    def get(self, deployment_id):
        """Get the details for a specific software deployment.

        :param deployment_id: ID of the software deployment
        """
        resp = self.client.get('/software_deployments/%s' % deployment_id)
        body = utils.get_response_body(resp)
        return SoftwareDeployment(self, body.get('software_deployment'))

    def create(self, **kwargs):
        """Create a software deployment."""
        resp = self.client.post('/software_deployments', data=kwargs)
        body = utils.get_response_body(resp)
        return SoftwareDeployment(self, body.get('software_deployment'))

    def update(self, deployment_id, **kwargs):
        """Update a software deployment."""
        resp = self.client.put('/software_deployments/%s' % deployment_id, data=kwargs)
        body = utils.get_response_body(resp)
        return SoftwareDeployment(self, body.get('software_deployment'))

    def delete(self, deployment_id):
        """Delete a software deployment."""
        self._delete('/software_deployments/%s' % deployment_id)