from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as composer_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetConfigConnectorIdentityForCluster(location, cluster_id, project_id):
    """Get Config Connector identity for the given cluster."""
    with composer_util.TemporaryKubeconfig(location, cluster_id):
        output = io.StringIO()
        composer_util.RunKubectlCommand(['get', 'ConfigConnectorContext', '-o', 'jsonpath="{.items[0].spec.googleServiceAccount}"'], out_func=output.write, err_func=log.err.write, namespace='config-control')
        identity = output.getvalue().replace('"', '')
        log.status.Print('Default Config Connector identity: [{identity}].\n\nFor example, to give Config Connector permission to manage Google Cloud resources in the same project:\ngcloud projects add-iam-policy-binding {project_id} \\\n    --member "serviceAccount:{identity}" \\\n    --role "roles/owner" \\\n    --project {project_id}\n'.format(identity=identity, project_id=project_id))