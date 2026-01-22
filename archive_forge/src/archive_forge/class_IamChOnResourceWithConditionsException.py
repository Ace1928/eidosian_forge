from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
class IamChOnResourceWithConditionsException(Exception):
    """Raised when trying to use "iam ch" on an IAM policy with conditions.

  Because the syntax for conditions is fairly complex, it doesn't make sense to
  specify them on the command line using a colon-delimited set of values in the
  same way you'd specify simple bindings - it would be a complex and potentially
  surprising interface, which isn't what you want when dealing with permissions.

  Additionally, providing partial functionality -- e.g. if a policy contains
  bindings with conditions, still allow users to interact with bindings that
  don't contain conditions -- might sound tempting, but results in a bad user
  experience. Bindings can be thought of as a mapping from (role, condition) ->
  [members]. Thus, a user might think they're editing the binding for (role1,
  condition1), but they'd really be editing the binding for (role1, None). Thus,
  we just raise an error if we encounter a binding with conditions present, and
  encourage users to use "iam {get,set}" instead.
  """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'IamChOnResourceWithConditionsException: %s' % self.message