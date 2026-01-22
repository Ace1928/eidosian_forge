from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
def MakeCondition(self, *args, **kwargs):
    if hasattr(self._messages, 'GoogleCloudRunV1Condition'):
        return self._messages.GoogleCloudRunV1Condition(*args, **kwargs)
    else:
        return getattr(self._messages, self.kind + 'Condition')(*args, **kwargs)