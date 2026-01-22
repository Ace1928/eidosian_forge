from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrimaryIdsValueValuesEnum(_messages.Enum):
    """Set `Action` for [`StudyInstanceUID`, `SeriesInstanceUID`,
    `SOPInstanceUID`, and `MediaStorageSOPInstanceUID`](http://dicom.nema.org/
    medical/dicom/2018e/output/chtml/part06/chapter_6.html).

    Values:
      PRIMARY_IDS_OPTION_UNSPECIFIED: No value provided. Default to the
        behavior specified by the base profile.
      KEEP: Keep primary IDs.
      REGEN: Regenerate primary IDs.
    """
    PRIMARY_IDS_OPTION_UNSPECIFIED = 0
    KEEP = 1
    REGEN = 2