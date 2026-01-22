from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BakExportOptionsValue(_messages.Message):
    """Options for exporting BAK files (SQL Server-only)

    Enums:
      BakTypeValueValuesEnum: Type of this bak file will be export, FULL or
        DIFF, SQL Server only

    Fields:
      bakType: Type of this bak file will be export, FULL or DIFF, SQL Server
        only
      copyOnly: Deprecated: copy_only is deprecated. Use differential_base
        instead
      differentialBase: Whether or not the backup can be used as a
        differential base copy_only backup can not be served as differential
        base
      stripeCount: Option for specifying how many stripes to use for the
        export. If blank, and the value of the striped field is true, the
        number of stripes is automatically chosen.
      striped: Whether or not the export should be striped.
    """

    class BakTypeValueValuesEnum(_messages.Enum):
        """Type of this bak file will be export, FULL or DIFF, SQL Server only

      Values:
        BAK_TYPE_UNSPECIFIED: Default type.
        FULL: Full backup.
        DIFF: Differential backup.
        TLOG: SQL Server Transaction Log
      """
        BAK_TYPE_UNSPECIFIED = 0
        FULL = 1
        DIFF = 2
        TLOG = 3
    bakType = _messages.EnumField('BakTypeValueValuesEnum', 1)
    copyOnly = _messages.BooleanField(2)
    differentialBase = _messages.BooleanField(3)
    stripeCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    striped = _messages.BooleanField(5)