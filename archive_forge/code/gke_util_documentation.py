from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
Get GKE cluster's full resource name and cluster URI.