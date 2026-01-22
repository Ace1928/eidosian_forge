from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
def _AddSubmitPredictionArgs(parser):
    """Add arguments for `jobs submit prediction` command."""
    parser.add_argument('job', help='Name of the batch prediction job.')
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model-dir', help='Cloud Storage location where the model files are located.')
    model_group.add_argument('--model', help='Name of the model to use for prediction.')
    parser.add_argument('--version', help='Model version to be used.\n\nThis flag may only be given if --model is specified. If unspecified, the default\nversion of the model will be used. To list versions for a model, run\n\n    $ gcloud ai-platform versions list\n')
    parser.add_argument('--input-paths', type=arg_parsers.ArgList(min_length=1), required=True, metavar='INPUT_PATH', help='Cloud Storage paths to the instances to run prediction on.\n\nWildcards (```*```) accepted at the *end* of a path. More than one path can be\nspecified if multiple file patterns are needed. For example,\n\n  gs://my-bucket/instances*,gs://my-bucket/other-instances1\n\nwill match any objects whose names start with `instances` in `my-bucket` as well\nas the `other-instances1` bucket, while\n\n  gs://my-bucket/instance-dir/*\n\nwill match any objects in the `instance-dir` "directory" (since directories\naren\'t a first-class Cloud Storage concept) of `my-bucket`.\n')
    jobs_util.DataFormatFlagMap().choice_arg.AddToParser(parser)
    parser.add_argument('--output-path', required=True, help='Cloud Storage path to which to save the output. Example: gs://my-bucket/output.')
    parser.add_argument('--region', required=True, help='The Compute Engine region to run the job in.')
    parser.add_argument('--max-worker-count', required=False, type=int, help='The maximum number of workers to be used for parallel processing. Defaults to 10 if not specified.')
    parser.add_argument('--batch-size', required=False, type=int, help='The number of records per batch. The service will buffer batch_size number of records in memory before invoking TensorFlow. Defaults to 64 if not specified.')
    flags.SIGNATURE_NAME.AddToParser(parser)
    flags.RUNTIME_VERSION.AddToParser(parser)
    labels_util.AddCreateLabelsFlags(parser)