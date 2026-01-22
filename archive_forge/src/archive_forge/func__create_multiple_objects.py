from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
def _create_multiple_objects(self):
    """Creates two versioned objects and return references to all versions.

    Returns:
      A four-tuple (a, b, a*, b*) of storage_uri.BucketStorageUri instances.
    """
    old_gsutil_object = self.CreateObject(bucket_uri=self.versioned_bucket, contents=b'foo')
    old_gsutil_object2 = self.CreateObject(bucket_uri=self.versioned_bucket, contents=b'bar')
    gsutil_object = self.CreateObject(bucket_uri=self.versioned_bucket, object_name=old_gsutil_object.object_name, contents=b'new_foo', gs_idempotent_generation=urigen(old_gsutil_object))
    gsutil_object2 = self.CreateObject(bucket_uri=self.versioned_bucket, object_name=old_gsutil_object2.object_name, contents=b'new_bar', gs_idempotent_generation=urigen(old_gsutil_object2))
    return (old_gsutil_object, old_gsutil_object2, gsutil_object, gsutil_object2)