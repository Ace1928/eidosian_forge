from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.code.cloud import cloud
from googlecloudsdk.core import yaml
import six
def SkaffoldConfig(self, service_file_path):
    """Generate the Skaffold yaml for the deploy."""
    skaffold_yaml = yaml.load(_SKAFFOLD_TEMPLATE)
    manifests = yaml_helper.GetOrCreate(skaffold_yaml, ('manifests', 'rawYaml'), constructor=list)
    manifests.append(service_file_path)
    artifact = {'image': self._settings.image}
    if isinstance(self._settings.builder, builders.BuildpackBuilder):
        artifact['buildpacks'] = {'builder': self._settings.builder.builder}
        artifact['sync'] = {'auto': False}
    else:
        dockerfile_rel_path = self._settings.builder.DockerfileRelPath(self._settings.context)
        artifact['docker'] = {'dockerfile': six.ensure_text(dockerfile_rel_path.encode('unicode_escape'))}
    artifacts = yaml_helper.GetOrCreate(skaffold_yaml, ('build', 'artifacts'), constructor=list)
    artifacts.append(artifact)
    skaffold_yaml['deploy']['cloudrun']['projectid'] = self._settings.project
    skaffold_yaml['deploy']['cloudrun']['region'] = self._settings.region
    if self._settings.local_port:
        port_forward_config = {'resourceType': 'service', 'resourceName': self._settings.service_name, 'port': 8080, 'localPort': self._settings.local_port}
        skaffold_yaml['portForward'] = [port_forward_config]
    return yaml.dump(skaffold_yaml)