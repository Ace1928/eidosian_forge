from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import notification_configuration_iterator
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
def _get_human_readable_notification(url, config):
    """Returns pretty notification string."""
    if config.custom_attributes:
        custom_attributes_string = '\n\tCustom attributes:'
        for attribute in config.custom_attributes.additionalProperties:
            custom_attributes_string += '\n\t\t{}: {}'.format(attribute.key, attribute.value)
    else:
        custom_attributes_string = ''
    if config.event_types or config.object_name_prefix:
        filters_string = '\n\tFilters:'
        if config.event_types:
            filters_string += '\n\t\tEvent Types: {}'.format(', '.join(config.event_types))
        if config.object_name_prefix:
            filters_string += "\n\t\tObject name prefix: '{}'".format(config.object_name_prefix)
    else:
        filters_string = ''
    return 'projects/_/buckets/{bucket}/notificationConfigs/{notification}\n\tCloud Pub/Sub topic: {topic}{custom_attributes}{filters}\n\n'.format(bucket=url.bucket_name, notification=config.id, topic=config.topic[_PUBSUB_DOMAIN_PREFIX_LENGTH:], custom_attributes=custom_attributes_string, filters=filters_string)