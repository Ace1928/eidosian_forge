from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1ArtifactsNpmPackage(_messages.Message):
    """Npm package to upload to Artifact Registry upon successful completion of
  all build steps.

  Fields:
    packagePath: Path to the package.json. e.g. workspace/path/to/package
    repository: Artifact Registry repository, in the form "https://$REGION-
      npm.pkg.dev/$PROJECT/$REPOSITORY" Npm package in the workspace specified
      by path will be zipped and uploaded to Artifact Registry with this
      location as a prefix.
  """
    packagePath = _messages.StringField(1)
    repository = _messages.StringField(2)