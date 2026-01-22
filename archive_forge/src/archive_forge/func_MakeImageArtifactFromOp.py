from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.cloudbuild import build
@classmethod
def MakeImageArtifactFromOp(cls, build_op):
    """Create Image BuildArtifact from build operation."""
    source = build.GetBuildProp(build_op, 'source')
    for prop in source.object_value.properties:
        if prop.key == 'storageSource':
            for storage_prop in prop.value.object_value.properties:
                if storage_prop.key == 'object':
                    image_name = storage_prop.value.string_value
    if image_name is None:
        raise build.BuildFailedError('Could not determine image name')
    return cls(cls.BuildType.IMAGE, image_name, build_op)