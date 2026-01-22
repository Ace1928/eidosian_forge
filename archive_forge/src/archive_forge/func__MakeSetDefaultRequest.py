from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import text
import six
def _MakeSetDefaultRequest(self, name):
    request = self.messages.GoogleCloudMlV1SetDefaultVersionRequest()
    return self.messages.MlProjectsModelsVersionsSetDefaultRequest(name=name, googleCloudMlV1SetDefaultVersionRequest=request)