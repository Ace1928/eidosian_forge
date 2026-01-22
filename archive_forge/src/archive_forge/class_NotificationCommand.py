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
class NotificationCommand(Command):
    """Implementation of gsutil notification command."""

    def _GetNotificationPathRegex(self):
        if not NotificationCommand._notification_path_regex:
            NotificationCommand._notification_path_regex = re.compile('/?(projects/[^/]+/)?b(uckets)?/(?P<bucket>[^/]+)/notificationConfigs/(?P<notification>[0-9]+)')
        return NotificationCommand._notification_path_regex
    _notification_path_regex = None
    command_spec = Command.CreateCommandSpec('notification', command_name_aliases=['notify', 'notifyconfig', 'notifications', 'notif'], usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='i:t:m:t:of:e:p:s', file_url_ok=False, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'watchbucket': [CommandArgument.MakeFreeTextArgument(), CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'stopchannel': [], 'list': [CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'delete': [CommandArgument.MakeZeroOrMoreCloudURLsArgument()], 'create': [CommandArgument.MakeFreeTextArgument(), CommandArgument.MakeNCloudBucketURLsArgument(1)]})
    help_spec = Command.HelpSpec(help_name='notification', help_name_aliases=['watchbucket', 'stopchannel', 'notifyconfig'], help_type='command_help', help_one_line_summary='Configure object change notification', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'create': _create_help_text, 'list': _list_help_text, 'delete': _delete_help_text, 'watchbucket': _watchbucket_help_text, 'stopchannel': _stopchannel_help_text})
    gcloud_storage_map = GcloudStorageMap(gcloud_command={'create': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'notifications', 'create'], flag_map={'-m': GcloudStorageFlag('--custom-attributes', repeat_type=shim_util.RepeatFlagType.DICT), '-e': GcloudStorageFlag('--event-types', repeat_type=shim_util.RepeatFlagType.LIST), '-p': GcloudStorageFlag('--object-prefix'), '-f': GcloudStorageFlag('--payload-format'), '-s': GcloudStorageFlag('--skip-topic-setup'), '-t': GcloudStorageFlag('--topic')}), 'delete': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'notifications', 'delete'], flag_map={}), 'list': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'notifications', 'list', '--human-readable'], flag_map={}, supports_output_translation=True)}, flag_map={})

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

    def _StopChannel(self):
        channel_id = self.args[0]
        resource_id = self.args[1]
        self.logger.info('Removing channel %s with resource identifier %s ...', channel_id, resource_id)
        self.gsutil_api.StopChannel(channel_id, resource_id, provider='gs')
        self.logger.info('Succesfully removed channel.')
        return 0

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

    def _CreateTopic(self, pubsub_topic, service_account):
        """Assures that a topic exists, creating it if necessary.

    Also adds GCS as a publisher on that bucket, if necessary.

    Args:
      pubsub_topic: name of the Cloud Pub/Sub topic to use/create.
      service_account: the GCS service account that needs publish permission.

    Returns:
      true if we modified IAM permissions, otherwise false.
    """
        pubsub_api = PubsubApi(logger=self.logger)
        try:
            pubsub_api.GetTopic(topic_name=pubsub_topic)
            self.logger.debug('Topic %s already exists', pubsub_topic)
        except NotFoundException:
            self.logger.debug('Creating topic %s', pubsub_topic)
            pubsub_api.CreateTopic(topic_name=pubsub_topic)
            self.logger.info('Created Cloud Pub/Sub topic %s', pubsub_topic)
        policy = pubsub_api.GetTopicIamPolicy(topic_name=pubsub_topic)
        binding = Binding(role='roles/pubsub.publisher', members=['serviceAccount:%s' % service_account])
        if binding not in policy.bindings:
            policy.bindings.append(binding)
            pubsub_api.SetTopicIamPolicy(topic_name=pubsub_topic, policy=policy)
            return True
        else:
            self.logger.debug('GCS already has publish permission to topic %s.', pubsub_topic)
            return False

    def _EnumerateNotificationsFromArgs(self, accept_notification_configs=True):
        """Yields bucket/notification tuples from command-line args.

    Given a list of strings that are bucket names (gs://foo) or notification
    config IDs, yield tuples of bucket names and their associated notifications.

    Args:
      accept_notification_configs: whether notification configs are valid args.
    Yields:
      Tuples of the form (bucket_name, Notification)
    """
        path_regex = self._GetNotificationPathRegex()
        for list_entry in self.args:
            match = path_regex.match(list_entry)
            if match:
                if not accept_notification_configs:
                    raise CommandException('%s %s accepts only bucket names, but you provided %s' % (self.command_name, self.subcommand_name, list_entry))
                bucket_name = match.group('bucket')
                notification_id = match.group('notification')
                found = False
                for notification in self.gsutil_api.ListNotificationConfigs(bucket_name, provider='gs'):
                    if notification.id == notification_id:
                        yield (bucket_name, notification)
                        found = True
                        break
                if not found:
                    raise NotFoundException('Could not find notification %s' % list_entry)
            else:
                storage_url = StorageUrlFromString(list_entry)
                if not storage_url.IsCloudUrl():
                    raise CommandException('The %s command must be used on cloud buckets or notification config names.' % self.command_name)
                if storage_url.scheme != 'gs':
                    raise CommandException('The %s command only works on gs:// buckets.')
                path = None
                if storage_url.IsProvider():
                    path = 'gs://*'
                elif storage_url.IsBucket():
                    path = list_entry
                if not path:
                    raise CommandException('The %s command cannot be used on cloud objects, only buckets' % self.command_name)
                for blr in self.WildcardIterator(path).IterBuckets(bucket_fields=['id']):
                    for notification in self.gsutil_api.ListNotificationConfigs(blr.storage_url.bucket_name, provider='gs'):
                        yield (blr.storage_url.bucket_name, notification)

    def _List(self):
        self.CheckArguments()
        if self.sub_opts:
            if '-o' in dict(self.sub_opts):
                for bucket_name in self.args:
                    self._ListChannels(bucket_name)
        else:
            for bucket_name, notification in self._EnumerateNotificationsFromArgs(accept_notification_configs=False):
                self._PrintNotificationDetails(bucket_name, notification)
        return 0

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

    def _Delete(self):
        for bucket_name, notification in self._EnumerateNotificationsFromArgs():
            self._DeleteNotification(bucket_name, notification.id)
        return 0

    def _DeleteNotification(self, bucket_name, notification_id):
        self.gsutil_api.DeleteNotificationConfig(bucket_name, notification=notification_id, provider='gs')
        return 0

    def _RunSubCommand(self, func):
        try:
            self.sub_opts, self.args = getopt.getopt(self.args, self.command_spec.supported_sub_args)
            metrics.LogCommandParams(sub_opts=self.sub_opts)
            return func(self)
        except getopt.GetoptError:
            self.RaiseInvalidArgumentException()
    SUBCOMMANDS = {'create': _Create, 'list': _List, 'delete': _Delete, 'watchbucket': _WatchBucket, 'stopchannel': _StopChannel}

    def RunCommand(self):
        """Command entry point for the notification command."""
        self.subcommand_name = self.args.pop(0)
        if self.subcommand_name in NotificationCommand.SUBCOMMANDS:
            metrics.LogCommandParams(subcommands=[self.subcommand_name])
            return self._RunSubCommand(NotificationCommand.SUBCOMMANDS[self.subcommand_name])
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.' % (self.subcommand_name, self.command_name))