from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UploadGenericArtifactRequest(_messages.Message):
    """The request to upload a generic artifact. The created GenericArtifact
  will have the resource name {parent}/genericArtifacts/package_id:version_id.
  The created file will have the resource name
  {parent}/files/package_id:version_id:filename.

  Fields:
    filename: The name of the file of the generic artifact to be uploaded.
      E.g. "example-file.zip" The filename should only include letters,
      numbers, and url safe characters, i.e. [a-zA-Z0-9-_.~@], and cannot
      exceed 64 characters.
    name: Deprecated. Use package_id, version_id and filename instead. The
      resource name of the generic artifact. E.g. "projects/math/locations/us/
      repositories/operations/genericArtifacts/addition/1.0.0/add.py"
    packageId: The ID of the package of the generic artifact. If the package
      does not exist, a new package will be created. E.g. "pkg-1" The
      package_id must start with a letter, end with a letter or number, only
      contain letters, numbers, and hyphens, i.e. [a-z0-9-], and cannot exceed
      64 characters.
    versionId: The ID of the version of the generic artifact. If the version
      does not exist, a new version will be created. E.g."1.0.0" The
      version_id must start and end with a letter or number, can only contain
      lowercase letters, numbers, hyphens and periods, i.e. [a-z0-9-.] and
      cannot exceed a total of 64 characters. While "latest" is a well-known
      name for the latest version of a package, it is not yet supported and is
      reserved for future use. Creating a version called "latest" is not
      allowed.
  """
    filename = _messages.StringField(1)
    name = _messages.StringField(2)
    packageId = _messages.StringField(3)
    versionId = _messages.StringField(4)