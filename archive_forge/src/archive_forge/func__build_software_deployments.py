import collections
from osc_lib.command import command
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient import exc
def _build_software_deployments(self, resources):
    """Build a dict of software deployments from the supplied resources.

        The key is the deployment ID.
        """
    df = {}
    if not resources:
        return df
    for r in resources.values():
        if r.resource_type not in ('OS::Heat::StructuredDeployment', 'OS::Heat::SoftwareDeployment'):
            continue
        try:
            sd = self.heat_client.software_deployments.get(deployment_id=r.physical_resource_id)
            df[r.physical_resource_id] = sd
        except exc.HTTPNotFound:
            pass
    return df