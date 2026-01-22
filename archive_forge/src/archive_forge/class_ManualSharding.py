from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManualSharding(_messages.Message):
    """Shards test cases into the specified groups of packages, classes, and/or
  methods. With manual sharding enabled, specifying test targets via
  environment_variables or in InstrumentationTest is invalid.

  Fields:
    testTargetsForShard: Required. Group of packages, classes, and/or test
      methods to be run for each manually-created shard. You must specify at
      least one shard if this field is present. When you select one or more
      physical devices, the number of repeated test_targets_for_shard must be
      <= 50. When you select one or more ARM virtual devices, it must be <=
      200. When you select only x86 virtual devices, it must be <= 500.
  """
    testTargetsForShard = _messages.MessageField('TestTargetsForShard', 1, repeated=True)