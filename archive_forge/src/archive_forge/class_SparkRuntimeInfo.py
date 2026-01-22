from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkRuntimeInfo(_messages.Message):
    """A SparkRuntimeInfo object.

  Fields:
    javaHome: A string attribute.
    javaVersion: A string attribute.
    scalaVersion: A string attribute.
  """
    javaHome = _messages.StringField(1)
    javaVersion = _messages.StringField(2)
    scalaVersion = _messages.StringField(3)