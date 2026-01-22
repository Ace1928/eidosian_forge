from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.endpoints import config_reporter
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.endpoints import services_util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six.moves.urllib.parse
class _BaseDeploy(object):
    """Create deploy base class for all release tracks."""

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go
          on the command line after this command. Positional arguments are
          allowed.
    """
        _CommonArgs(parser)
        parser.add_argument('--validate-only', action='store_true', help='If included, the command validates the service configuration(s), but does not deploy them. The service must exist in order to validate the configuration(s).')
        parser.add_argument('--force', '-f', action='store_true', default=False, help='Force the deployment even if any hazardous changes to the service configuration are detected.')

    def MakeConfigFileMessage(self, file_contents, filename, file_type):
        """Constructs a ConfigFile message from a config file.

    Args:
      file_contents: The contents of the config file.
      filename: The full path to the config file.
      file_type: FileTypeValueValuesEnum describing the type of config file.

    Returns:
      The constructed ConfigFile message.
    """
        messages = services_util.GetMessagesModule()
        file_types = messages.ConfigFile.FileTypeValueValuesEnum
        if file_type != file_types.FILE_DESCRIPTOR_SET_PROTO:
            file_contents = http_encoding.Encode(file_contents)
        return messages.ConfigFile(fileContents=file_contents, filePath=os.path.basename(filename), fileType=file_type)

    def ShowConfigReport(self, service, service_config_id, log_func=log.warning):
        """Run and display results (if any) from the Push Advisor.

    Args:
      service: The name of the service for which to compare configs.
      service_config_id: The new config ID to compare against the active config.
      log_func: The function to which to pass advisory messages
        (default: log.warning).

    Returns:
      The number of advisory messages returned by the Push Advisor.
    """
        num_changes_with_advice = 0
        reporter = config_reporter.ConfigReporter(service)
        reporter.new_config.SetConfigId(service_config_id)
        reporter.old_config.SetConfigUseDefaultId()
        change_report = reporter.RunReport()
        if not change_report or not change_report.configChanges:
            return 0
        changes = change_report.configChanges
        for change in changes:
            if change.advices:
                if num_changes_with_advice < NUM_ADVICE_TO_PRINT:
                    log_func('%s\n', services_util.PushAdvisorConfigChangeToString(change))
                num_changes_with_advice += 1
        if num_changes_with_advice > NUM_ADVICE_TO_PRINT:
            log_func('%s total changes with advice found, check config report file for full list.\n', num_changes_with_advice)
        return num_changes_with_advice

    def CheckPushAdvisor(self, unused_force=False):
        """Run the Push Advisor and return whether the command should abort.

    Args:
      unused_force: bool, unused in the default implementation.

    Returns:
      True if the deployment should be aborted due to warnings, otherwise
      False if it's safe to continue.
    """
        return False

    def AttemptToEnableService(self, service_name, is_async):
        """Attempt to enable a service. If lacking permission, log a warning.

    Args:
      service_name: The service to enable.
      is_async: If true, return immediately instead of waiting for the operation
          to finish.
    """
        project_id = properties.VALUES.core.project.Get(required=True)
        try:
            enable_api.EnableService(project_id, service_name, is_async)
            log.status.Print('\n')
        except services_exceptions.EnableServicePermissionDeniedException:
            log.warning('Attempted to enable service [{0}] on project [{1}], but did not have required permissions. Please ensure this service is enabled before using your Endpoints service.\n'.format(service_name, project_id))

    def Run(self, args):
        """Run 'endpoints services deploy'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
          with.

    Returns:
      The response from the Update API call.

    Raises:
      BadFileExceptionn: if the provided service configuration files are
          invalid or cannot be read.
    """
        messages = services_util.GetMessagesModule()
        client = services_util.GetClientInstance()
        file_types = messages.ConfigFile.FileTypeValueValuesEnum
        self.service_name = self.service_version = config_contents = None
        config_files = []
        self.validate_only = args.validate_only
        give_proto_deprecate_warning = False
        if not self.validate_only and (not args.IsSpecified('format')):
            args.format = 'none'
        for service_config_file in args.service_config_file:
            config_contents = services_util.ReadServiceConfigFile(service_config_file)
            if services_util.FilenameMatchesExtension(service_config_file, ['.json', '.yaml', '.yml']):
                service_config_dict = services_util.LoadJsonOrYaml(config_contents)
                if not service_config_dict:
                    raise calliope_exceptions.BadFileException('Could not read JSON or YAML from service config file [{0}].'.format(service_config_file))
                if 'swagger' in service_config_dict:
                    if 'host' not in service_config_dict:
                        raise calliope_exceptions.BadFileException('Malformed input. Found Swagger service config in file [{}], but no host was specified. Add a host specification to the config file.'.format(service_config_file))
                    if not self.service_name and service_config_dict.get('host'):
                        self.service_name = service_config_dict.get('host')
                    config_files.append(self.MakeConfigFileMessage(config_contents, service_config_file, file_types.OPEN_API_YAML))
                elif service_config_dict.get('type') == 'google.api.Service':
                    if not self.service_name and service_config_dict.get('name'):
                        self.service_name = service_config_dict.get('name')
                    config_files.append(self.MakeConfigFileMessage(config_contents, service_config_file, file_types.SERVICE_CONFIG_YAML))
                elif 'name' in service_config_dict:
                    if len(args.service_config_file) > 1:
                        raise calliope_exceptions.BadFileException('Ambiguous input. Found normalized service configuration in file [{0}], but received multiple input files. To upload normalized service config, please provide it separately from other input files to avoid ambiguity.'.format(service_config_file))
                    if self.validate_only:
                        raise exceptions.InvalidFlagError('The --validate-only flag is not supported when using normalized service configs as input.')
                    self.service_name = service_config_dict.get('name')
                    config_files = []
                    break
                else:
                    raise calliope_exceptions.BadFileException('Unable to parse Open API, or Google Service Configuration specification from {0}'.format(service_config_file))
            elif services_util.IsProtoDescriptor(service_config_file):
                config_files.append(self.MakeConfigFileMessage(config_contents, service_config_file, file_types.FILE_DESCRIPTOR_SET_PROTO))
            elif services_util.IsRawProto(service_config_file):
                give_proto_deprecate_warning = True
                config_files.append(self.MakeConfigFileMessage(config_contents, service_config_file, file_types.PROTO_FILE))
            else:
                raise calliope_exceptions.BadFileException('Could not determine the content type of file [{0}]. Supported extensions are .json .yaml .yml .pb and .descriptor'.format(service_config_file))
        if give_proto_deprecate_warning:
            log.warning('Support for uploading uncompiled .proto files is deprecated and will soon be removed. Use compiled descriptor sets (.pb) instead.\n')
        was_service_created = False
        if not services_util.DoesServiceExist(self.service_name):
            project_id = properties.VALUES.core.project.Get(required=True)
            if self.validate_only:
                if not console_io.CanPrompt():
                    raise exceptions.InvalidConditionError(VALIDATE_NEW_ERROR.format(service_name=self.service_name, project_id=project_id))
                if not console_io.PromptContinue(VALIDATE_NEW_PROMPT.format(service_name=self.service_name, project_id=project_id)):
                    return None
            services_util.CreateService(self.service_name, project_id)
            was_service_created = True
        if config_files:
            push_config_result = services_util.PushMultipleServiceConfigFiles(self.service_name, config_files, args.async_, validate_only=self.validate_only)
            self.service_config_id = services_util.GetServiceConfigIdFromSubmitConfigSourceResponse(push_config_result)
        else:
            push_config_result = services_util.PushNormalizedGoogleServiceConfig(self.service_name, properties.VALUES.core.project.Get(required=True), services_util.LoadJsonOrYaml(config_contents))
            self.service_config_id = push_config_result.id
        if not self.service_config_id:
            raise exceptions.InvalidConditionError('Failed to retrieve Service Configuration Id.')
        if self.CheckPushAdvisor(args.force):
            return None
        if not self.validate_only:
            percentages = messages.TrafficPercentStrategy.PercentagesValue()
            percentages.additionalProperties.append(messages.TrafficPercentStrategy.PercentagesValue.AdditionalProperty(key=self.service_config_id, value=100.0))
            traffic_percent_strategy = messages.TrafficPercentStrategy(percentages=percentages)
            rollout = messages.Rollout(serviceName=self.service_name, trafficPercentStrategy=traffic_percent_strategy)
            rollout_create = messages.ServicemanagementServicesRolloutsCreateRequest(rollout=rollout, serviceName=self.service_name)
            rollout_operation = client.services_rollouts.Create(rollout_create)
            services_util.ProcessOperationResult(rollout_operation, args.async_)
            if was_service_created:
                self.AttemptToEnableService(self.service_name, args.async_)
        return push_config_result

    def Epilog(self, resources_were_displayed):
        if resources_were_displayed and (not self.validate_only):
            log.status.Print('Service Configuration [{0}] uploaded for service [{1}]\n'.format(self.service_config_id, self.service_name))
            management_url = GenerateManagementUrl(self.service_name)
            log.status.Print('To manage your API, go to: ' + management_url)