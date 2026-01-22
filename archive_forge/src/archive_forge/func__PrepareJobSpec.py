from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.custom_jobs import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import validation as common_validation
from googlecloudsdk.command_lib.ai.custom_jobs import custom_jobs_util
from googlecloudsdk.command_lib.ai.custom_jobs import flags
from googlecloudsdk.command_lib.ai.custom_jobs import validation
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _PrepareJobSpec(self, args, api_client, project):
    job_config = api_client.ImportResourceMessage(args.config, 'CustomJobSpec') if args.config else api_client.GetMessage('CustomJobSpec')()
    validation.ValidateCreateArgs(args, job_config, self._version)
    worker_pool_specs = list(custom_jobs_util.UpdateWorkerPoolSpecsIfLocalPackageRequired(args.worker_pool_spec or [], args.display_name, project))
    job_spec = custom_jobs_util.ConstructCustomJobSpec(api_client, base_config=job_config, worker_pool_specs=worker_pool_specs, network=args.network, service_account=args.service_account, enable_web_access=args.enable_web_access, enable_dashboard_access=args.enable_dashboard_access, args=args.args, command=args.command, python_package_uri=args.python_package_uris, persistent_resource_id=args.persistent_resource_id)
    return job_spec