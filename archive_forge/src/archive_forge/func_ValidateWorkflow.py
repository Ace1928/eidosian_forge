from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
def ValidateWorkflow(workflow, first_deployment=False):
    if first_deployment and (not workflow.sourceContents):
        raise exceptions.RequiredArgumentException('--source', 'required on first deployment')