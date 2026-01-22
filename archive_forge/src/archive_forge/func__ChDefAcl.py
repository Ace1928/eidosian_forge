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