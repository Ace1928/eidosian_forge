from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import getopt
import re
import time
import uuid
from datetime import datetime
from gslib import metrics
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PublishPermissionDeniedException
from gslib.command import Command
from gslib.command import NO_MAX
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.help_provider import CreateHelpText
from gslib.project_id import PopulateProjectId
from gslib.pubsub_api import PubsubApi
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.pubsub_apitools.pubsub_v1_messages import Binding
from gslib.utils import copy_helper
from gslib.utils import shim_util
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
def _PrintNotificationDetails(self, bucket, notification):
    print('projects/_/buckets/{bucket}/notificationConfigs/{notification}\n\tCloud Pub/Sub topic: {topic}'.format(bucket=bucket, notification=notification.id, topic=notification.topic[len('//pubsub.googleapis.com/'):]))
    if notification.custom_attributes:
        print('\tCustom attributes:')
        for attr in notification.custom_attributes.additionalProperties:
            print('\t\t%s: %s' % (attr.key, attr.value))
    filters = []
    if notification.event_types:
        filters.append('\t\tEvent Types: %s' % ', '.join(notification.event_types))
    if notification.object_name_prefix:
        filters.append("\t\tObject name prefix: '%s'" % notification.object_name_prefix)
    if filters:
        print('\tFilters:')
        for line in filters:
            print(line)
    self.logger.info('')