from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def GetApplicationRef(args):
    """Returns a application reference."""
    app_ref = args.CONCEPTS.application.Parse()
    if not app_ref.Name():
        raise exceptions.InvalidArgumentException('application', 'application id must be non-empty.')
    return app_ref