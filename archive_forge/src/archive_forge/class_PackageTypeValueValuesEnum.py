from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PackageTypeValueValuesEnum(_messages.Enum):
    """The type of package: os, maven, go, etc.

    Values:
      PACKAGE_TYPE_UNSPECIFIED: <no description>
      OS: Operating System
      MAVEN: Java packages from Maven.
      GO: Go third-party packages.
      GO_STDLIB: Go toolchain + standard library packages.
      PYPI: Python packages.
      NPM: NPM packages.
      NUGET: Nuget (C#/.NET) packages.
      RUBYGEMS: Ruby packges (from RubyGems package manager).
      RUST: Rust packages from Cargo (Github ecosystem is `RUST`).
      COMPOSER: PHP packages from Composer package manager.
    """
    PACKAGE_TYPE_UNSPECIFIED = 0
    OS = 1
    MAVEN = 2
    GO = 3
    GO_STDLIB = 4
    PYPI = 5
    NPM = 6
    NUGET = 7
    RUBYGEMS = 8
    RUST = 9
    COMPOSER = 10