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
def _Create(self):
    self.CheckArguments()
    pubsub_topic = None
    payload_format = None
    custom_attributes = {}
    event_types = []
    object_name_prefix = None
    should_setup_topic = True
    if self.sub_opts:
        for o, a in self.sub_opts:
            if o == '-e':
                event_types.append(a)
            elif o == '-f':
                payload_format = a
            elif o == '-m':
                if ':' not in a:
                    raise CommandException('Custom attributes specified with -m should be of the form key:value')
                key, value = a.split(':', 1)
                custom_attributes[key] = value
            elif o == '-p':
                object_name_prefix = a
            elif o == '-s':
                should_setup_topic = False
            elif o == '-t':
                pubsub_topic = a
    if payload_format not in PAYLOAD_FORMAT_MAP:
        raise CommandException("Must provide a payload format with -f of either 'json' or 'none'")
    payload_format = PAYLOAD_FORMAT_MAP[payload_format]
    bucket_arg = self.args[-1]
    bucket_url = StorageUrlFromString(bucket_arg)
    if not bucket_url.IsCloudUrl() or not bucket_url.IsBucket():
        raise CommandException("%s %s requires a GCS bucket name, but got '%s'" % (self.command_name, self.subcommand_name, bucket_arg))
    if bucket_url.scheme != 'gs':
        raise CommandException('The %s command can only be used with gs:// bucket URLs.' % self.command_name)
    bucket_name = bucket_url.bucket_name
    self.logger.debug('Creating notification for bucket %s', bucket_url)
    bucket_metadata = self.gsutil_api.GetBucket(bucket_name, fields=['projectNumber'], provider=bucket_url.scheme)
    bucket_project_number = bucket_metadata.projectNumber
    if not pubsub_topic:
        pubsub_topic = 'projects/%s/topics/%s' % (PopulateProjectId(None), bucket_name)
    if not pubsub_topic.startswith('projects/'):
        pubsub_topic = 'projects/%s/topics/%s' % (PopulateProjectId(None), pubsub_topic)
    self.logger.debug('Using Cloud Pub/Sub topic %s', pubsub_topic)
    just_modified_topic_permissions = False
    if should_setup_topic:
        service_account = self.gsutil_api.GetProjectServiceAccount(bucket_project_number, provider=bucket_url.scheme).email_address
        self.logger.debug('Service account for project %d: %s', bucket_project_number, service_account)
        just_modified_topic_permissions = self._CreateTopic(pubsub_topic, service_account)
    for attempt_number in range(0, 2):
        try:
            create_response = self.gsutil_api.CreateNotificationConfig(bucket_name, pubsub_topic=pubsub_topic, payload_format=payload_format, custom_attributes=custom_attributes, event_types=event_types if event_types else None, object_name_prefix=object_name_prefix, provider=bucket_url.scheme)
            break
        except PublishPermissionDeniedException:
            if attempt_number == 0 and just_modified_topic_permissions:
                self.logger.info('Retrying create notification in 10 seconds (new permissions may take up to 10 seconds to take effect.)')
                time.sleep(10)
            else:
                raise
    notification_name = 'projects/_/buckets/%s/notificationConfigs/%s' % (bucket_name, create_response.id)
    self.logger.info('Created notification config %s', notification_name)
    return 0