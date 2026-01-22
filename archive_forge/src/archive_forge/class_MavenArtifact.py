from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MavenArtifact(_messages.Message):
    """A Maven artifact to upload to Artifact Registry upon successful
  completion of all build steps.

  Fields:
    artifactId: Maven `artifactId` value used when uploading the artifact to
      Artifact Registry.
    groupId: Maven `groupId` value used when uploading the artifact to
      Artifact Registry.
    path: Path to an artifact in the build's workspace to be uploaded to
      Artifact Registry. This can be either an absolute path, e.g.
      /workspace/my-app/target/my-app-1.0.SNAPSHOT.jar or a relative path from
      /workspace, e.g. my-app/target/my-app-1.0.SNAPSHOT.jar.
    repository: Artifact Registry repository, in the form "https://$REGION-
      maven.pkg.dev/$PROJECT/$REPOSITORY" Artifact in the workspace specified
      by path will be uploaded to Artifact Registry with this location as a
      prefix.
    version: Maven `version` value used when uploading the artifact to
      Artifact Registry.
  """
    artifactId = _messages.StringField(1)
    groupId = _messages.StringField(2)
    path = _messages.StringField(3)
    repository = _messages.StringField(4)
    version = _messages.StringField(5)