from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
import operator
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import build as app_cloud_build
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import exceptions
from googlecloudsdk.api_lib.app import instances_util
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app import region_util
from googlecloudsdk.api_lib.app import service_util
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.third_party.appengine.admin.tools.conversion import convert_yaml
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
def _CreateVersionResource(self, service_config, manifest, version_id, build, extra_config_settings=None, service_account_email=None):
    """Constructs a Version resource for deployment.

    Args:
      service_config: ServiceYamlInfo, Service info parsed from a service yaml
        file.
      manifest: Dictionary mapping source files to Google Cloud Storage
        locations.
      version_id: str, The version of the service.
      build: BuildArtifact, The build ID, image path, or build options.
      extra_config_settings: dict, client config settings to pass to the server
        as beta settings.
      service_account_email: identity of this deployed version. If not set,
        Admin API will fallback to use the App Engine default appspot SA.

    Returns:
      A Version resource whose Deployment includes either a container pointing
        to a completed image, or a build pointing to an in-progress build.
    """
    config_dict = copy.deepcopy(service_config.parsed.ToDict())
    if 'entrypoint' not in config_dict:
        config_dict['entrypoint'] = ''
    try:
        schema_parser = convert_yaml.GetSchemaParser(self.client._VERSION)
        json_version_resource = schema_parser.ConvertValue(config_dict)
    except ValueError as e:
        raise exceptions.ConfigError('[{f}] could not be converted to the App Engine configuration format for the following reason: {msg}'.format(f=service_config.file, msg=six.text_type(e)))
    log.debug('Converted YAML to JSON: "{0}"'.format(json.dumps(json_version_resource, indent=2, sort_keys=True)))
    if service_account_email is not None:
        json_version_resource['serviceAccount'] = service_account_email
    json_version_resource['deployment'] = {}
    json_version_resource['deployment']['files'] = manifest
    if build:
        if build.IsImage():
            json_version_resource['deployment']['container'] = {'image': build.identifier}
        elif build.IsBuildId():
            json_version_resource['deployment']['build'] = {'cloudBuildId': build.identifier}
        elif build.IsBuildOptions():
            json_version_resource['deployment']['cloudBuildOptions'] = build.identifier
    version_resource = encoding.PyValueToMessage(self.messages.Version, json_version_resource)
    if version_resource.envVariables:
        version_resource.envVariables.additionalProperties.sort(key=lambda x: x.key)
    if extra_config_settings:
        if 'betaSettings' not in json_version_resource:
            json_version_resource['betaSettings'] = {}
        json_version_resource['betaSettings'].update(extra_config_settings)
    if 'betaSettings' in json_version_resource:
        json_dict = json_version_resource.get('betaSettings')
        attributes = []
        for key, value in sorted(json_dict.items()):
            attributes.append(self.messages.Version.BetaSettingsValue.AdditionalProperty(key=key, value=value))
        version_resource.betaSettings = self.messages.Version.BetaSettingsValue(additionalProperties=attributes)
    try:
        version_resource.deployment.files.additionalProperties.sort(key=operator.attrgetter('key'))
    except AttributeError:
        pass
    version_resource.id = version_id
    return version_resource