from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import flags as shared_flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.ai.custom_jobs import custom_jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddLocalRunCustomJobFlags(parser):
    """Add local-run related flags to the parser."""
    application_group = parser.add_mutually_exclusive_group()
    application_group.add_argument('--python-module', metavar='PYTHON_MODULE', help=textwrap.dedent("\n      Name of the python module to execute, in 'trainer.train' or 'train'\n      format. Its path should be relative to the `work_dir`.\n      "))
    application_group.add_argument('--script', metavar='SCRIPT', help=textwrap.dedent('\n      The relative path of the file to execute. Accepets a Python file or an\n      arbitrary bash script. This path should be relative to the `work_dir`.\n      '))
    parser.add_argument('--local-package-path', metavar='LOCAL_PATH', suggestion_aliases=['--work-dir'], help=textwrap.dedent('\n      local path of the directory where the python-module or script exists.\n      If not specified, it use the directory where you run the this command.\n\n      Only the contents of this directory will be accessible to the built\n      container image.\n      '))
    parser.add_argument('--extra-dirs', metavar='EXTRA_DIR', type=arg_parsers.ArgList(), help=textwrap.dedent('\n      Extra directories under the working directory to include, besides the one\n      that contains the main executable.\n\n      By default, only the parent directory of the main script or python module\n      is copied to the container.\n      For example, if the module is "training.task" or the script is\n      "training/task.py", the whole "training" directory, including its\n      sub-directories, will always be copied to the container. You may specify\n      this flag to also copy other directories if necessary.\n\n      Note: if no parent is specified in \'python_module\' or \'scirpt\', the whole\n      working directory is copied, then you don\'t need to specify this flag.\n      '))
    parser.add_argument('--executor-image-uri', metavar='IMAGE_URI', required=True, suggestion_aliases=['--base-image'], help=textwrap.dedent('\n      URI or ID of the container image in either the Container Registry or local\n      that will run the application.\n      See https://cloud.google.com/vertex-ai/docs/training/pre-built-containers\n      for available pre-built container images provided by Vertex AI for training.\n      '))
    parser.add_argument('--requirements', metavar='REQUIREMENTS', type=arg_parsers.ArgList(), help=textwrap.dedent('\n      Python dependencies from PyPI to be used when running the application.\n      If this is not specified, and there is no "setup.py" or "requirements.txt"\n      in the working directory, your application will only have access to what\n      exists in the base image with on other dependencies.\n\n      Example:\n      \'tensorflow-cpu, pandas==1.2.0, matplotlib>=3.0.2\'\n      '))
    parser.add_argument('--extra-packages', metavar='PACKAGE', type=arg_parsers.ArgList(), help=textwrap.dedent("\n      Local paths to Python archives used as training dependencies in the image\n      container.\n      These can be absolute or relative paths. However, they have to be under\n      the work_dir; Otherwise, this tool will not be able to access it.\n\n      Example:\n      'dep1.tar.gz, ./downloads/dep2.whl'\n      "))
    parser.add_argument('--output-image-uri', metavar='OUTPUT_IMAGE', help=textwrap.dedent('\n      Uri of the custom container image to be built with the your application\n      packed in.\n      '))
    parser.add_argument('--gpu', action='store_true', default=False, help='Enable to use GPU.')
    parser.add_argument('--docker-run-options', metavar='DOCKER_RUN_OPTIONS', hidden=True, type=arg_parsers.ArgList(), help=textwrap.dedent("\n      Custom Docker run options to pass to image during execution.\n      For example, '--no-healthcheck, -a stdin'.\n\n      See https://docs.docker.com/engine/reference/commandline/run/#options for\n      more details.\n      "))
    parser.add_argument('--service-account-key-file', metavar='ACCOUNT_KEY_FILE', help=textwrap.dedent("\n      The JSON file of a Google Cloud service account private key.\n      When specified, the corresponding service account will be used to\n      authenticate the local container to access Google Cloud services.\n      Note that the key file won't be copied to the container, it will be\n      mounted during running time.\n      "))
    parser.add_argument('args', nargs=argparse.REMAINDER, default=[], help='Additional user arguments to be forwarded to your application.', example='$ {command} --script=my_run.sh --base-image=gcr.io/my/image -- --my-arg bar --enable_foo')