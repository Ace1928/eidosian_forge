from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def AddReadinessProbe(self):
    self._AddFlag('--readiness-probe', default=False, action='store_true', hidden=True, help='Add a readiness probe to the list of containers that delays deployment stabilization until the application app has bound to $PORT')