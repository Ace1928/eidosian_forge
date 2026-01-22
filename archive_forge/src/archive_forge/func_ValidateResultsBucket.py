from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def ValidateResultsBucket(args):
    """Do some basic sanity checks on the format of the results-bucket arg.

  Args:
    args: the argparse.Namespace containing all the args for the command.

  Raises:
    InvalidArgumentException: the bucket name is not valid or includes objects.
  """
    if args.results_bucket is None:
        return
    try:
        bucket_ref = storage_util.BucketReference.FromArgument(args.results_bucket, require_prefix=False)
    except Exception as err:
        raise exceptions.InvalidArgumentException('results-bucket', six.text_type(err))
    args.results_bucket = bucket_ref.bucket