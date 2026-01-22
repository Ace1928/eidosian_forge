from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RelationshipNote(_messages.Message):
    """RelationshipNote represents an SPDX Relationship section:
  https://spdx.github.io/spdx-spec/7-relationships-between-SPDX-elements/

  Enums:
    TypeValueValuesEnum: The type of relationship between the source and
      target SPDX elements

  Fields:
    type: The type of relationship between the source and target SPDX elements
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of relationship between the source and target SPDX elements

    Values:
      RELATIONSHIP_TYPE_UNSPECIFIED: Unspecified
      DESCRIBES: Is to be used when SPDXRef-DOCUMENT describes SPDXRef-A
      DESCRIBED_BY: Is to be used when SPDXRef-A is described by SPDXREF-
        Document
      CONTAINS: Is to be used when SPDXRef-A contains SPDXRef-B
      CONTAINED_BY: Is to be used when SPDXRef-A is contained by SPDXRef-B
      DEPENDS_ON: Is to be used when SPDXRef-A depends on SPDXRef-B
      DEPENDENCY_OF: Is to be used when SPDXRef-A is dependency of SPDXRef-B
      DEPENDENCY_MANIFEST_OF: Is to be used when SPDXRef-A is a manifest file
        that lists a set of dependencies for SPDXRef-B
      BUILD_DEPENDENCY_OF: Is to be used when SPDXRef-A is a build dependency
        of SPDXRef-B
      DEV_DEPENDENCY_OF: Is to be used when SPDXRef-A is a development
        dependency of SPDXRef-B
      OPTIONAL_DEPENDENCY_OF: Is to be used when SPDXRef-A is an optional
        dependency of SPDXRef-B
      PROVIDED_DEPENDENCY_OF: Is to be used when SPDXRef-A is a to be provided
        dependency of SPDXRef-B
      TEST_DEPENDENCY_OF: Is to be used when SPDXRef-A is a test dependency of
        SPDXRef-B
      RUNTIME_DEPENDENCY_OF: Is to be used when SPDXRef-A is a dependency
        required for the execution of SPDXRef-B
      EXAMPLE_OF: Is to be used when SPDXRef-A is an example of SPDXRef-B
      GENERATES: Is to be used when SPDXRef-A generates SPDXRef-B
      GENERATED_FROM: Is to be used when SPDXRef-A was generated from
        SPDXRef-B
      ANCESTOR_OF: Is to be used when SPDXRef-A is an ancestor (same lineage
        but pre-dates) SPDXRef-B
      DESCENDANT_OF: Is to be used when SPDXRef-A is a descendant of (same
        lineage but postdates) SPDXRef-B
      VARIANT_OF: Is to be used when SPDXRef-A is a variant of (same lineage
        but not clear which came first) SPDXRef-B
      DISTRIBUTION_ARTIFACT: Is to be used when distributing SPDXRef-A
        requires that SPDXRef-B also be distributed
      PATCH_FOR: Is to be used when SPDXRef-A is a patch file for (to be
        applied to) SPDXRef-B
      PATCH_APPLIED: Is to be used when SPDXRef-A is a patch file that has
        been applied to SPDXRef-B
      COPY_OF: Is to be used when SPDXRef-A is an exact copy of SPDXRef-B
      FILE_ADDED: Is to be used when SPDXRef-A is a file that was added to
        SPDXRef-B
      FILE_DELETED: Is to be used when SPDXRef-A is a file that was deleted
        from SPDXRef-B
      FILE_MODIFIED: Is to be used when SPDXRef-A is a file that was modified
        from SPDXRef-B
      EXPANDED_FROM_ARCHIVE: Is to be used when SPDXRef-A is expanded from the
        archive SPDXRef-B
      DYNAMIC_LINK: Is to be used when SPDXRef-A dynamically links to
        SPDXRef-B
      STATIC_LINK: Is to be used when SPDXRef-A statically links to SPDXRef-B
      DATA_FILE_OF: Is to be used when SPDXRef-A is a data file used in
        SPDXRef-B
      TEST_CASE_OF: Is to be used when SPDXRef-A is a test case used in
        testing SPDXRef-B
      BUILD_TOOL_OF: Is to be used when SPDXRef-A is used to build SPDXRef-B
      DEV_TOOL_OF: Is to be used when SPDXRef-A is used as a development tool
        for SPDXRef-B
      TEST_OF: Is to be used when SPDXRef-A is used for testing SPDXRef-B
      TEST_TOOL_OF: Is to be used when SPDXRef-A is used as a test tool for
        SPDXRef-B
      DOCUMENTATION_OF: Is to be used when SPDXRef-A provides documentation of
        SPDXRef-B
      OPTIONAL_COMPONENT_OF: Is to be used when SPDXRef-A is an optional
        component of SPDXRef-B
      METAFILE_OF: Is to be used when SPDXRef-A is a metafile of SPDXRef-B
      PACKAGE_OF: Is to be used when SPDXRef-A is used as a package as part of
        SPDXRef-B
      AMENDS: Is to be used when (current) SPDXRef-DOCUMENT amends the SPDX
        information in SPDXRef-B
      PREREQUISITE_FOR: Is to be used when SPDXRef-A is a prerequisite for
        SPDXRef-B
      HAS_PREREQUISITE: Is to be used when SPDXRef-A has as a prerequisite
        SPDXRef-B
      OTHER: Is to be used for a relationship which has not been defined in
        the formal SPDX specification. A description of the relationship
        should be included in the Relationship comments field
    """
        RELATIONSHIP_TYPE_UNSPECIFIED = 0
        DESCRIBES = 1
        DESCRIBED_BY = 2
        CONTAINS = 3
        CONTAINED_BY = 4
        DEPENDS_ON = 5
        DEPENDENCY_OF = 6
        DEPENDENCY_MANIFEST_OF = 7
        BUILD_DEPENDENCY_OF = 8
        DEV_DEPENDENCY_OF = 9
        OPTIONAL_DEPENDENCY_OF = 10
        PROVIDED_DEPENDENCY_OF = 11
        TEST_DEPENDENCY_OF = 12
        RUNTIME_DEPENDENCY_OF = 13
        EXAMPLE_OF = 14
        GENERATES = 15
        GENERATED_FROM = 16
        ANCESTOR_OF = 17
        DESCENDANT_OF = 18
        VARIANT_OF = 19
        DISTRIBUTION_ARTIFACT = 20
        PATCH_FOR = 21
        PATCH_APPLIED = 22
        COPY_OF = 23
        FILE_ADDED = 24
        FILE_DELETED = 25
        FILE_MODIFIED = 26
        EXPANDED_FROM_ARCHIVE = 27
        DYNAMIC_LINK = 28
        STATIC_LINK = 29
        DATA_FILE_OF = 30
        TEST_CASE_OF = 31
        BUILD_TOOL_OF = 32
        DEV_TOOL_OF = 33
        TEST_OF = 34
        TEST_TOOL_OF = 35
        DOCUMENTATION_OF = 36
        OPTIONAL_COMPONENT_OF = 37
        METAFILE_OF = 38
        PACKAGE_OF = 39
        AMENDS = 40
        PREREQUISITE_FOR = 41
        HAS_PREREQUISITE = 42
        OTHER = 43
    type = _messages.EnumField('TypeValueValuesEnum', 1)