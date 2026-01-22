from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddFromRunContainerFlag():
    return base.Argument('--from-run-container', hidden=True, help='\n          The container name, which Cloud Deploy will use to\n          generate a CloudRun manifest.yaml and a skaffold.yaml file.\n          The generated Skaffold file and manifest file will be\n          available in the Google Cloud Storage source staging directory\n          after the release is complete.\n      ')