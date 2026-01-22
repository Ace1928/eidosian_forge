from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import enum
from googlecloudsdk.command_lib.survey import question
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import pkg_resources
def _GetSurveyContentDirectory():
    """Get the directory containing all surveys in yaml format.

  Returns:
    Path to the surveys directory, i.e.
      $CLOUDSDKROOT/lib/googlecloudsdk/command_lib/survey/contents
  """
    return os.path.join(os.path.dirname(encoding.Decode(__file__)), 'contents')