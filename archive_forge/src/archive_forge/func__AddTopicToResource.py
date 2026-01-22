from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddTopicToResource(topic_name, new_config, resource, resource_name):
    """Add the PubsubConfig message to Repo/ProjectConfig message."""
    if resource.pubsubConfigs is None:
        config_additional_properties = []
    else:
        config_additional_properties = resource.pubsubConfigs.additionalProperties
    resource_msg_module = _MESSAGES.ProjectConfig
    if resource_name == 'repo':
        resource_msg_module = _MESSAGES.Repo
    config_additional_properties.append(resource_msg_module.PubsubConfigsValue.AdditionalProperty(key=topic_name, value=new_config))
    return resource_msg_module(name=resource.name, pubsubConfigs=resource_msg_module.PubsubConfigsValue(additionalProperties=config_additional_properties))