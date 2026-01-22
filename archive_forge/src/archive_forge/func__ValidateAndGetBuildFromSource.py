import enum
import os.path
from googlecloudsdk.api_lib.run import api_enabler
from googlecloudsdk.api_lib.run import k8s_object
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
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
def _ValidateAndGetBuildFromSource(self, containers):
    build_from_source = {name: container for name, container in containers.items() if not container.IsSpecified('image')}
    if len(build_from_source) > 1:
        needs_image = [name for name, container in build_from_source.items() if not flags.FlagIsExplicitlySet(container, 'source')]
        if needs_image:
            raise exceptions.RequiredImageArgumentException(needs_image)
        raise c_exceptions.InvalidArgumentException('--container', 'At most one container can be deployed from source.')
    for name, container in build_from_source.items():
        if not flags.FlagIsExplicitlySet(container, 'source'):
            if console_io.CanPrompt():
                container.source = flags.PromptForDefaultSource(name)
            else:
                if name:
                    message = 'Container {} requires a container image to deploy (e.g. `gcr.io/cloudrun/hello:latest`) if no build source is provided.'.format(name)
                else:
                    message = 'Requires a container image to deploy (e.g. `gcr.io/cloudrun/hello:latest`) if no build source is provided.'
                raise c_exceptions.RequiredArgumentException('--image', message)
    return build_from_source