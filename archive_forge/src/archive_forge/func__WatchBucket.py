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
def _WatchBucket(self):
    """Creates a watch on a bucket given in self.args."""
    self.CheckArguments()
    identifier = None
    client_token = None
    if self.sub_opts:
        for o, a in self.sub_opts:
            if o == '-i':
                identifier = a
            if o == '-t':
                client_token = a
    identifier = identifier or str(uuid.uuid4())
    watch_url = self.args[0]
    bucket_arg = self.args[-1]
    if not watch_url.lower().startswith('https://'):
        raise CommandException('The application URL must be an https:// URL.')
    bucket_url = StorageUrlFromString(bucket_arg)
    if not (bucket_url.IsBucket() and bucket_url.scheme == 'gs'):
        raise CommandException('The %s command can only be used with gs:// bucket URLs.' % self.command_name)
    if not bucket_url.IsBucket():
        raise CommandException('URL must name a bucket for the %s command.' % self.command_name)
    self.logger.info('Watching bucket %s with application URL %s ...', bucket_url, watch_url)
    try:
        channel = self.gsutil_api.WatchBucket(bucket_url.bucket_name, watch_url, identifier, token=client_token, provider=bucket_url.scheme)
    except AccessDeniedException as e:
        self.logger.warn(NOTIFICATION_AUTHORIZATION_FAILED_MESSAGE.format(watch_error=str(e), watch_url=watch_url))
        raise
    channel_id = channel.id
    resource_id = channel.resourceId
    client_token = channel.token
    self.logger.info('Successfully created watch notification channel.')
    self.logger.info('Watch channel identifier: %s', channel_id)
    self.logger.info('Canonicalized resource identifier: %s', resource_id)
    self.logger.info('Client state token: %s', client_token)
    return 0