from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PackageInfoNote(_messages.Message):
    """PackageInfoNote represents an SPDX Package Information section:
  https://spdx.github.io/spdx-spec/3-package-information/

  Fields:
    analyzed: Indicates whether the file content of this package has been
      available for or subjected to analysis when creating the SPDX document
    attribution: A place for the SPDX data creator to record, at the package
      level, acknowledgements that may be needed to be communicated in some
      contexts
    checksum: Provide an independently reproducible mechanism that permits
      unique identification of a specific package that correlates to the data
      in this SPDX file
    copyright: Identify the copyright holders of the package, as well as any
      dates present
    detailedDescription: A more detailed description of the package
    downloadLocation: This section identifies the download Universal Resource
      Locator (URL), or a specific location within a version control system
      (VCS) for the package at the time that the SPDX file was created
    externalRefs: ExternalRef
    filesLicenseInfo: Contain the license the SPDX file creator has concluded
      as governing the This field is to contain a list of all licenses found
      in the package. The relationship between licenses (i.e., conjunctive,
      disjunctive) is not specified in this field \\u2013 it is simply a
      listing of all licenses found
    homePage: Provide a place for the SPDX file creator to record a web site
      that serves as the package's home page
    licenseDeclared: List the licenses that have been declared by the authors
      of the package
    originator: If the package identified in the SPDX file originated from a
      different person or organization than identified as Package Supplier,
      this field identifies from where or whom the package originally came
    packageType: The type of package: OS, MAVEN, GO, GO_STDLIB, etc.
    summaryDescription: A short description of the package
    supplier: Identify the actual distribution source for the
      package/directory identified in the SPDX file
    title: Identify the full name of the package as given by the Package
      Originator
    verificationCode: This field provides an independently reproducible
      mechanism identifying specific contents of a package based on the actual
      files (except the SPDX file itself, if it is included in the package)
      that make up each package and that correlates to the data in this SPDX
      file
    version: Identify the version of the package
  """
    analyzed = _messages.BooleanField(1)
    attribution = _messages.StringField(2)
    checksum = _messages.StringField(3)
    copyright = _messages.StringField(4)
    detailedDescription = _messages.StringField(5)
    downloadLocation = _messages.StringField(6)
    externalRefs = _messages.MessageField('ExternalRef', 7, repeated=True)
    filesLicenseInfo = _messages.StringField(8, repeated=True)
    homePage = _messages.StringField(9)
    licenseDeclared = _messages.MessageField('License', 10)
    originator = _messages.StringField(11)
    packageType = _messages.StringField(12)
    summaryDescription = _messages.StringField(13)
    supplier = _messages.StringField(14)
    title = _messages.StringField(15)
    verificationCode = _messages.StringField(16)
    version = _messages.StringField(17)