from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpriteSheet(_messages.Message):
    """Sprite sheet configuration.

  Fields:
    columnCount: The maximum number of sprites per row in a sprite sheet. The
      default is 0, which indicates no maximum limit.
    endTimeOffset: End time in seconds, relative to the output file timeline.
      When `end_time_offset` is not specified, the sprites are generated until
      the end of the output file.
    filePrefix: Required. File name prefix for the generated sprite sheets.
      Each sprite sheet has an incremental 10-digit zero-padded suffix
      starting from 0 before the extension, such as
      `sprite_sheet0000000123.jpeg`.
    format: Format type. The default is `jpeg`. Supported formats: - `jpeg`
    interval: Starting from `0s`, create sprites at regular intervals. Specify
      the interval value in seconds.
    quality: The quality of the generated sprite sheet. Enter a value between
      1 and 100, where 1 is the lowest quality and 100 is the highest quality.
      The default is 100. A high quality value corresponds to a low image data
      compression ratio.
    rowCount: The maximum number of rows per sprite sheet. When the sprite
      sheet is full, a new sprite sheet is created. The default is 0, which
      indicates no maximum limit.
    spriteHeightPixels: Required. The height of sprite in pixels. Must be an
      even integer. To preserve the source aspect ratio, set the
      SpriteSheet.sprite_height_pixels field or the
      SpriteSheet.sprite_width_pixels field, but not both (the API will
      automatically calculate the missing field). For portrait videos that
      contain horizontal ASR and rotation metadata, provide the height, in
      pixels, per the horizontal ASR. The API calculates the width per the
      horizontal ASR. The API detects any rotation metadata and swaps the
      requested height and width for the output.
    spriteWidthPixels: Required. The width of sprite in pixels. Must be an
      even integer. To preserve the source aspect ratio, set the
      SpriteSheet.sprite_width_pixels field or the
      SpriteSheet.sprite_height_pixels field, but not both (the API will
      automatically calculate the missing field). For portrait videos that
      contain horizontal ASR and rotation metadata, provide the width, in
      pixels, per the horizontal ASR. The API calculates the height per the
      horizontal ASR. The API detects any rotation metadata and swaps the
      requested height and width for the output.
    startTimeOffset: Start time in seconds, relative to the output file
      timeline. Determines the first sprite to pick. The default is `0s`.
    totalCount: Total number of sprites. Create the specified number of
      sprites distributed evenly across the timeline of the output media. The
      default is 100.
  """
    columnCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    endTimeOffset = _messages.StringField(2)
    filePrefix = _messages.StringField(3)
    format = _messages.StringField(4)
    interval = _messages.StringField(5)
    quality = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    rowCount = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    spriteHeightPixels = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    spriteWidthPixels = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    startTimeOffset = _messages.StringField(10)
    totalCount = _messages.IntegerField(11, variant=_messages.Variant.INT32)