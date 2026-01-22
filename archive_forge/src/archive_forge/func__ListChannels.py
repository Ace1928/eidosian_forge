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
def _ListChannels(self, bucket_arg):
    """Lists active channel watches on a bucket given in self.args."""
    bucket_url = StorageUrlFromString(bucket_arg)
    if not (bucket_url.IsBucket() and bucket_url.scheme == 'gs'):
        raise CommandException('The %s command can only be used with gs:// bucket URLs.' % self.command_name)
    if not bucket_url.IsBucket():
        raise CommandException('URL must name a bucket for the %s command.' % self.command_name)
    channels = self.gsutil_api.ListChannels(bucket_url.bucket_name, provider='gs').items
    self.logger.info('Bucket %s has the following active Object Change Notifications:', bucket_url.bucket_name)
    for idx, channel in enumerate(channels):
        self.logger.info('\tNotification channel %d:', idx + 1)
        self.logger.info('\t\tChannel identifier: %s', channel.channel_id)
        self.logger.info('\t\tResource identifier: %s', channel.resource_id)
        self.logger.info('\t\tApplication URL: %s', channel.push_url)
        self.logger.info('\t\tCreated by: %s', channel.subscriber_email)
        self.logger.info('\t\tCreation time: %s', str(datetime.fromtimestamp(channel.creation_time_ms / 1000)))
    return 0