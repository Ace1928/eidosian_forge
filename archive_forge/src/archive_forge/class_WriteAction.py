from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WriteAction(_messages.Message):
    """Create or modify a file.

  Enums:
    ModeValueValuesEnum: The new mode of the file.

  Fields:
    contents: The new contents of the file.
    mode: The new mode of the file.
    path: The path of the file to write.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """The new mode of the file.

    Values:
      FILE_MODE_UNSPECIFIED: No file mode was specified.
      NORMAL: Neither a symbolic link nor executable.
      SYMLINK: A symbolic link.
      EXECUTABLE: An executable.
    """
        FILE_MODE_UNSPECIFIED = 0
        NORMAL = 1
        SYMLINK = 2
        EXECUTABLE = 3
    contents = _messages.BytesField(1)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)
    path = _messages.StringField(3)