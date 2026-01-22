from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import os
import re
import subprocess
import textwrap
import six
from six.moves import zip
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite.messages import DecodeError
from boto import config
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import GetFailureCount
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import IamChOnResourceWithConditionsException
from gslib.help_provider import CreateHelpText
from gslib.metrics import LogCommandParams
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.storage_url import GetSchemeFromUrlString
from gslib.storage_url import IsKnownUrlScheme
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreMixOfBucketsAndObjects
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import IAM_POLICY_VERSION
from gslib.utils.constants import NO_MAX
from gslib.utils import iam_helper
from gslib.utils.iam_helper import BindingStringToTuple
from gslib.utils.iam_helper import BindingsTuple
from gslib.utils.iam_helper import DeserializeBindingsTuple
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.iam_helper import SerializeBindingsTuple
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.shim_util import GcloudStorageFlag
@Retry(PreconditionException, tries=3, timeout_secs=1.0)
def _PatchIamHelperInternal(self, storage_url, bindings_tuples, thread_state=None):
    policy = self.GetIamHelper(storage_url, thread_state=thread_state)
    etag, bindings = (policy.etag, policy.bindings)
    for binding in bindings:
        if binding.condition:
            message = 'Could not patch IAM policy for %s.' % storage_url
            message += '\n'
            message += '\n'.join(textwrap.wrap('The resource had conditions present in its IAM policy bindings, which is not supported by "iam ch". %s' % IAM_CH_CONDITIONS_WORKAROUND_MSG))
            raise IamChOnResourceWithConditionsException(message)
    orig_bindings = list(bindings)
    for is_grant, diff in bindings_tuples:
        bindings_dict = iam_helper.BindingsMessageToUpdateDict(bindings)
        diff_dict = iam_helper.BindingsMessageToUpdateDict(diff)
        new_bindings_dict = PatchBindings(bindings_dict, diff_dict, is_grant)
        bindings = [apitools_messages.Policy.BindingsValueListEntry(role=r, members=list(m)) for r, m in six.iteritems(new_bindings_dict)]
    if IsEqualBindings(bindings, orig_bindings):
        self.logger.info('No changes made to %s', storage_url)
        return
    policy = apitools_messages.Policy(bindings=bindings, etag=etag)
    self._SetIamHelperInternal(storage_url, policy, thread_state=thread_state)