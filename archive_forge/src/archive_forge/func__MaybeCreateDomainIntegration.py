import enum
import os.path
from googlecloudsdk.api_lib.run import api_enabler
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import traffic
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import container_parser
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import resource_change_validators
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.integrations import run_apps_operations
from googlecloudsdk.command_lib.run.sourcedeploys import builders
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
def _MaybeCreateDomainIntegration(self, service, args):
    domain_name = self._MaybeGetDomain(args)
    if not domain_name:
        return
    first_region = flags.GetMultiRegion(args).split(',')[0]
    service_name = service.metadata.name
    params = {'set-mapping': '%s/*:%s' % (domain_name, service_name)}
    pretty_print.Info('Mapping multi-region Service {svc} to domain {domain}', svc=service_name, domain=domain_name)
    with run_apps_operations.ConnectWithRegion(first_region, None, self.ReleaseTrack()) as stacks_client:
        stacks_client.VerifyLocation()
        if stacks_client.MaybeGetIntegrationGeneric('custom-domains', 'router'):
            stacks_client.UpdateIntegration('custom-domains', params)
            pretty_print.Success('Sucessfully updated mapping {svc} to domain {domain}', svc=service_name, domain=domain_name)
        else:
            stacks_client.CreateIntegration('custom-domains', params, None)
            pretty_print.Success('Sucessfully created mapping {svc} to domain {domain}', svc=service_name, domain=domain_name)