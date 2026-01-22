from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def GetBinauthzEvaluationMode(args):
    evaluation_mode = getattr(args, 'binauthz_evaluation_mode', None)
    if evaluation_mode is None:
        return None
    return _BINAUTHZ_EVAL_MODE_ENUM_MAPPER.GetEnumForChoice(_ToHyphenCase(evaluation_mode))