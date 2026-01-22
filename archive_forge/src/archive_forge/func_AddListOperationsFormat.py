from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def AddListOperationsFormat(parser):
    parser.display_info.AddTransforms({'operationState': _TransformOperationState, 'operationTimestamp': _TransformOperationTimestamp, 'operationType': _TransformOperationType, 'operationWarnings': _TransformOperationWarnings})
    parser.display_info.AddFormat("table(name.segment():label=NAME, metadata.operationTimestamp():label=TIMESTAMP,metadata.operationType():label=TYPE, metadata.operationState():label=STATE, status.code.yesno(no=''):label=ERROR, metadata.operationWarnings():label=WARNINGS)")