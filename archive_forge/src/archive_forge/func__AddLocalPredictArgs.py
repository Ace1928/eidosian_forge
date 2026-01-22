from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import local_utils
from googlecloudsdk.command_lib.ml_engine import predict_utilities
from googlecloudsdk.core import log
def _AddLocalPredictArgs(parser):
    """Add arguments for `gcloud ai-platform local predict` command."""
    parser.add_argument('--model-dir', required=True, help='Path to the model.')
    flags.FRAMEWORK_MAPPER.choice_arg.AddToParser(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--json-request', help='      Path to a local file containing the body of JSON request.\n\n      An example of a JSON request:\n\n          {\n            "instances": [\n              {"x": [1, 2], "y": [3, 4]},\n              {"x": [-1, -2], "y": [-3, -4]}\n            ]\n          }\n\n      This flag accepts "-" for stdin.\n      ')
    group.add_argument('--json-instances', help='      Path to a local file from which instances are read.\n      Instances are in JSON format; newline delimited.\n\n      An example of the JSON instances file:\n\n          {"images": [0.0, ..., 0.1], "key": 3}\n          {"images": [0.0, ..., 0.1], "key": 2}\n          ...\n\n      This flag accepts "-" for stdin.\n      ')
    group.add_argument('--text-instances', help='      Path to a local file from which instances are read.\n      Instances are in UTF-8 encoded text format; newline delimited.\n\n      An example of the text instances file:\n\n          107,4.9,2.5,4.5,1.7\n          100,5.7,2.8,4.1,1.3\n          ...\n\n      This flag accepts "-" for stdin.\n      ')
    flags.SIGNATURE_NAME.AddToParser(parser)