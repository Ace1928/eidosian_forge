from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib import gcs_json_api
from gslib import metrics
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import Preconditions
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import SetAclExceptionHandler
from gslib.command import SetAclFuncWrapper
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.help_provider import CreateHelpText
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import acl_helper
from gslib.utils.constants import NO_MAX
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import PRIVATE_DEFAULT_OBJ_ACL
class DefAclCommand(Command):
    """Implementation of gsutil defacl command."""
    command_spec = Command.CreateCommandSpec('defacl', command_name_aliases=['setdefacl', 'getdefacl', 'chdefacl'], usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='fg:u:d:p:', file_url_ok=False, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'set': [CommandArgument.MakeFileURLOrCannedACLArgument(), CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'get': [CommandArgument.MakeNCloudBucketURLsArgument(1)], 'ch': [CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()]})
    help_spec = Command.HelpSpec(help_name='defacl', help_name_aliases=['default acl', 'setdefacl', 'getdefacl', 'chdefacl'], help_type='command_help', help_one_line_summary='Get, set, or change default ACL on buckets', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text, 'ch': _ch_help_text})

    def get_gcloud_storage_args(self):
        sub_command = self.args.pop(0)
        if sub_command == 'get':
            gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'buckets', 'describe', '--format=multi(defaultObjectAcl:format=json)', '--raw'], flag_map={})
        elif sub_command == 'set':
            acl_file_or_predefined_acl = self.args.pop(0)
            if os.path.isfile(acl_file_or_predefined_acl):
                gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--default-object-acl-file=' + acl_file_or_predefined_acl], flag_map={})
            else:
                if acl_file_or_predefined_acl in gcs_json_api.FULL_PREDEFINED_ACL_XML_TO_JSON_TRANSLATION:
                    predefined_acl = gcs_json_api.FULL_PREDEFINED_ACL_XML_TO_JSON_TRANSLATION[acl_file_or_predefined_acl]
                else:
                    predefined_acl = acl_file_or_predefined_acl
                gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--predefined-default-object-acl=' + predefined_acl], flag_map={})
        elif sub_command == 'ch':
            self.ParseSubOpts()
            self.sub_opts = acl_helper.translate_sub_opts_for_shim(self.sub_opts)
            gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update'], flag_map={'-g': GcloudStorageFlag('--add-default-object-acl-grant'), '-p': GcloudStorageFlag('--add-default-object-acl-grant'), '-u': GcloudStorageFlag('--add-default-object-acl-grant'), '-d': GcloudStorageFlag('--remove-default-object-acl-grant'), '-f': GcloudStorageFlag('--continue-on-error')})
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def _CalculateUrlsStartArg(self):
        if not self.args:
            self.RaiseWrongNumberOfArgumentsException()
        if self.args[0].lower() == 'set' or self.command_alias_used == 'setdefacl':
            return 1
        else:
            return 0

    def _SetDefAcl(self):
        if not StorageUrlFromString(self.args[-1]).IsBucket():
            raise CommandException('URL must name a bucket for the %s command' % self.command_name)
        try:
            self.SetAclCommandHelper(SetAclFuncWrapper, SetAclExceptionHandler)
        except AccessDeniedException:
            self._WarnServiceAccounts()
            raise

    def _GetDefAcl(self):
        if not StorageUrlFromString(self.args[0]).IsBucket():
            raise CommandException('URL must name a bucket for the %s command' % self.command_name)
        self.GetAndPrintAcl(self.args[0])

    def _ChDefAcl(self):
        """Parses options and changes default object ACLs on specified buckets."""
        self.parse_versions = True
        self.changes = []
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-g':
                    self.changes.append(acl_helper.AclChange(a, scope_type=acl_helper.ChangeType.GROUP))
                if o == '-u':
                    self.changes.append(acl_helper.AclChange(a, scope_type=acl_helper.ChangeType.USER))
                if o == '-p':
                    self.changes.append(acl_helper.AclChange(a, scope_type=acl_helper.ChangeType.PROJECT))
                if o == '-d':
                    self.changes.append(acl_helper.AclDel(a))
        if not self.changes:
            raise CommandException('Please specify at least one access change with the -g, -u, or -d flags')
        if not UrlsAreForSingleProvider(self.args) or StorageUrlFromString(self.args[0]).scheme != 'gs':
            raise CommandException('The "{0}" command can only be used with gs:// URLs'.format(self.command_name))
        bucket_urls = set()
        for url_arg in self.args:
            for result in self.WildcardIterator(url_arg):
                if not result.storage_url.IsBucket():
                    raise CommandException('The defacl ch command can only be applied to buckets.')
                bucket_urls.add(result.storage_url)
        for storage_url in bucket_urls:
            self.ApplyAclChanges(storage_url)

    @Retry(ServiceException, tries=3, timeout_secs=1)
    def ApplyAclChanges(self, url):
        """Applies the changes in self.changes to the provided URL."""
        bucket = self.gsutil_api.GetBucket(url.bucket_name, provider=url.scheme, fields=['defaultObjectAcl', 'metageneration'])
        current_acl = bucket.defaultObjectAcl
        if self._ApplyAclChangesAndReturnChangeCount(url, current_acl) == 0:
            self.logger.info('No changes to %s', url)
            return
        if not current_acl:
            current_acl.append(PRIVATE_DEFAULT_OBJ_ACL)
        try:
            preconditions = Preconditions(meta_gen_match=bucket.metageneration)
            bucket_metadata = apitools_messages.Bucket(defaultObjectAcl=current_acl)
            self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, preconditions=preconditions, provider=url.scheme, fields=['id'])
            self.logger.info('Updated default ACL on %s', url)
        except BadRequestException as e:
            raise CommandException('Received bad request from server: %s' % str(e))
        except AccessDeniedException:
            self._WarnServiceAccounts()
            raise CommandException('Failed to set acl for %s. Please ensure you have OWNER-role access to this resource.' % url)

    def _ApplyAclChangesAndReturnChangeCount(self, storage_url, defacl_message):
        modification_count = 0
        for change in self.changes:
            modification_count += change.Execute(storage_url, defacl_message, 'defacl', self.logger)
        return modification_count

    def RunCommand(self):
        """Command entry point for the defacl command."""
        action_subcommand = self.args.pop(0)
        self.ParseSubOpts(check_args=True)
        self.def_acl = True
        self.continue_on_error = False
        if action_subcommand == 'get':
            func = self._GetDefAcl
        elif action_subcommand == 'set':
            func = self._SetDefAcl
        elif action_subcommand in ('ch', 'change'):
            func = self._ChDefAcl
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.\nSee "gsutil help defacl".' % (action_subcommand, self.command_name))
        metrics.LogCommandParams(subcommands=[action_subcommand], sub_opts=self.sub_opts)
        func()
        return 0