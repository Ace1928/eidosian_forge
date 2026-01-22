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
def PatchIamHelper(self, storage_url, bindings_tuples, thread_state=None):
    """Patches an IAM policy for a single, resolved bucket / object URL.

    The patch is applied by altering the policy from an IAM get request, and
    setting the new IAM with the specified etag. Because concurrent IAM set
    requests may alter the etag, we may need to retry this operation several
    times before success.

    Args:
      storage_url: A CloudUrl instance with no wildcards, pointing to a
                   specific bucket or object.
      bindings_tuples: A list of BindingsTuple instances.
      thread_state: CloudApiDelegator instance which is passed from
                    command.WorkerThread.__init__() if the -m flag is
                    specified. Will use self.gsutil_api if thread_state is set
                    to None.
    """
    try:
        self._PatchIamHelperInternal(storage_url, bindings_tuples, thread_state=thread_state)
    except ServiceException:
        if self.continue_on_error:
            self.everything_set_okay = False
        else:
            raise
    except IamChOnResourceWithConditionsException as e:
        if self.continue_on_error:
            self.everything_set_okay = False
            self.tried_ch_on_resource_with_conditions = True
            self.logger.debug(e.message)
        else:
            raise CommandException(e.message)