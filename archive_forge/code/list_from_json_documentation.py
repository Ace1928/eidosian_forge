from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
Read JSON data and list it on the standard output.

  *{command}* is a test harness for resource output formatting and filtering.
  It behaves like any other `gcloud ... list` command except that the resources
  are read from a JSON data file.

  The input JSON data is either a single resource object or a list of resource
  objects of the same type. The resources are printed on the standard output.
  The default output format is *json*.
  