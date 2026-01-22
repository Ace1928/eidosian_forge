from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Intents(base.Group):
    """Create, list, describe, and delete Dialogflow intents.

  Intents convert a number of user expressions or patterns into an action. An
  action is an extraction of a user command or sentence semantics.
  """