from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScreenshotCluster(_messages.Message):
    """A ScreenshotCluster object.

  Fields:
    activity: A string that describes the activity of every screen in the
      cluster.
    clusterId: A unique identifier for the cluster. @OutputOnly
    keyScreen: A singular screen that represents the cluster as a whole. This
      screen will act as the "cover" of the entire cluster. When users look at
      the clusters, only the key screen from each cluster will be shown. Which
      screen is the key screen is determined by the ClusteringAlgorithm
    screens: Full list of screens.
  """
    activity = _messages.StringField(1)
    clusterId = _messages.StringField(2)
    keyScreen = _messages.MessageField('Screen', 3)
    screens = _messages.MessageField('Screen', 4, repeated=True)