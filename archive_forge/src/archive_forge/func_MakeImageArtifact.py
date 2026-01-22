from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.cloudbuild import build
@classmethod
def MakeImageArtifact(cls, image_name):
    return cls(cls.BuildType.IMAGE, image_name)