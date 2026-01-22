from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def GetImageFromFile(path):
    """Builds a GooglePrivacyDlpV2ByteContentItem message from a path.

  Will attempt to set message.type from file extension (if present).

  Args:
    path: the path arg given to the command.

  Raises:
    ImageFileError: if the image path does not exist and does not have a valid
    extension.

  Returns:
    GooglePrivacyDlpV2ByteContentItem: an message containing image data for
    the API on the image to analyze.
  """
    extension = os.path.splitext(path)[-1].lower()
    extension = extension or 'n_a'
    image_item = _GetMessageClass('GooglePrivacyDlpV2ByteContentItem')
    if os.path.isfile(path) and _ValidateExtension(extension):
        enum_val = arg_utils.ChoiceToEnum(VALID_IMAGE_EXTENSIONS[extension], image_item.TypeValueValuesEnum)
        image = image_item(data=files.ReadBinaryFileContents(path), type=enum_val)
    else:
        raise ImageFileError('The image path [{}] does not exist or has an invalid extension. Must be one of [jpg, jpeg, png, bmp or svg]. Please double-check your input and try again.'.format(path))
    return image