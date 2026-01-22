from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import matrix_creator_common
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six
def _BuildShardingOption(self):
    """Build a ShardingOption for an AndroidInstrumentationTest."""
    if getattr(self._args, 'num_uniform_shards', {}):
        return self._messages.ShardingOption(uniformSharding=self._messages.UniformSharding(numShards=self._args.num_uniform_shards))
    elif getattr(self._args, 'test_targets_for_shard', {}):
        return self._messages.ShardingOption(manualSharding=self._BuildManualShard(self._args.test_targets_for_shard))