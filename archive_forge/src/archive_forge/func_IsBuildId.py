from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.cloudbuild import build
def IsBuildId(self):
    return self.build_type == self.BuildType.BUILD_ID