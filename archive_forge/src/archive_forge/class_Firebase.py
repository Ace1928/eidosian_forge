from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Firebase(base.Group):
    """Work with Google Firebase.

  To view all options available for using Firebase Test Lab, run:

      $ {command} test --help
  """
    category = base.MOBILE_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()