from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib import redis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.console import console_io
import six
def InstanceUpdateLabelsFlags():
    remove_group = base.ArgumentGroup(mutex=True)
    remove_group.AddArgument(labels_util.GetClearLabelsFlag())
    remove_group.AddArgument(labels_util.GetRemoveLabelsFlag(''))
    return [labels_util.GetUpdateLabelsFlag(''), remove_group]