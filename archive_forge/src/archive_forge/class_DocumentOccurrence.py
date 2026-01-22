from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentOccurrence(_messages.Message):
    """DocumentOccurrence represents an SPDX Document Creation Information
  section: https://spdx.github.io/spdx-spec/2-document-creation-information/

  Fields:
    createTime: Identify when the SPDX file was originally created. The date
      is to be specified according to combined date and time in UTC format as
      specified in ISO 8601 standard
    creatorComment: A field for creators of the SPDX file to provide general
      comments about the creation of the SPDX file or any other relevant
      comment not included in the other fields
    creators: Identify who (or what, in the case of a tool) created the SPDX
      file. If the SPDX file was created by an individual, indicate the
      person's name
    documentComment: A field for creators of the SPDX file content to provide
      comments to the consumers of the SPDX document
    externalDocumentRefs: Identify any external SPDX documents referenced
      within this SPDX document
    id: Identify the current SPDX document which may be referenced in
      relationships by other files, packages internally and documents
      externally
    licenseListVersion: A field for creators of the SPDX file to provide the
      version of the SPDX License List used when the SPDX file was created
    namespace: Provide an SPDX document specific namespace as a unique
      absolute Uniform Resource Identifier (URI) as specified in RFC-3986,
      with the exception of the '#' delimiter
    title: Identify name of this document as designated by creator
  """
    createTime = _messages.StringField(1)
    creatorComment = _messages.StringField(2)
    creators = _messages.StringField(3, repeated=True)
    documentComment = _messages.StringField(4)
    externalDocumentRefs = _messages.StringField(5, repeated=True)
    id = _messages.StringField(6)
    licenseListVersion = _messages.StringField(7)
    namespace = _messages.StringField(8)
    title = _messages.StringField(9)