from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FilterProfileValueValuesEnum(_messages.Enum):
    """Tag filtering profile that determines the tags to keep or remove.

    Values:
      TAG_FILTER_PROFILE_UNSPECIFIED: No tag filtration profile provided. Same
        as KEEP_ALL_PROFILE.
      MINIMAL_KEEP_LIST_PROFILE: Keep only tags required to produce valid
        DICOM.
      ATTRIBUTE_CONFIDENTIALITY_BASIC_PROFILE: Remove tags based on DICOM
        Standard's [Attribute Confidentiality Basic Profile (DICOM Standard
        Edition 2018e)] (http://dicom.nema.org/medical/dicom/2018e/output/chtm
        l/part15/chapter_E.html).
      KEEP_ALL_PROFILE: Keep all tags.
      DEIDENTIFY_TAG_CONTENTS: Inspects within tag contents (including tags
        nested in a sequence) and replaces sensitive text. The process can be
        configured using the TextConfig. Applies to all tags with the
        following Value Representation names: AE, LO, LT, PN, SH, ST, UC, UT,
        DA, DT, AS.
    """
    TAG_FILTER_PROFILE_UNSPECIFIED = 0
    MINIMAL_KEEP_LIST_PROFILE = 1
    ATTRIBUTE_CONFIDENTIALITY_BASIC_PROFILE = 2
    KEEP_ALL_PROFILE = 3
    DEIDENTIFY_TAG_CONTENTS = 4