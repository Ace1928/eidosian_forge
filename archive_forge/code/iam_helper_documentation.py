from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
from collections import namedtuple
import six
from apitools.base.protorpclite import protojson
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
Parses an iam ch bind string to a list of binding tuples.

  Args:
    is_grant: If true, binding is to be appended to IAM policy; else, delete
              this binding from the policy.
    input_str: A string representing a member-role binding.
               e.g. user:foo@bar.com:objectAdmin
                    user:foo@bar.com:objectAdmin,objectViewer
                    user:foo@bar.com
                    allUsers
                    deleted:user:foo@bar.com?uid=123:objectAdmin,objectViewer
                    deleted:serviceAccount:foo@bar.com?uid=123

  Raises:
    CommandException in the case of invalid input.

  Returns:
    A BindingsDictTuple instance.
  