from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as functions_api_util
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTerraformBlueprintFlag(parser):
    """Add TerraformBlueprint related flags."""
    input_values_help_text = 'Input variable values for the Terraform blueprint. It only\n      accepts (key, value) pairs where value is a scalar value.\n\nExamples:\n\nPass input values on command line:\n\n  $ {command} projects/p1/location/us-central1/deployments/my-deployment --gcs-source="gs://my-bucket" --input-values=projects=p1,region=r\n'
    inputs_file_help_text = 'A .tfvars file containing terraform variable values. --inputs-file flag is supported for python version 3.6 and above.\n\nExamples:\n\nPass input values on the command line:\n\n  $ {command} projects/p1/location/us-central1/deployments/my-deployment --gcs-source="gs://my-bucket" --inputs-file=path-to-tfvar-file.tfvar\n'
    gcs_source_help_text = 'URI of an object in Google Cloud Storage.\n      e.g. `gs://{bucket}/{object}`\n\nExamples:\n\nCreate a deployment from a storage my-bucket:\n\n  $ {command} projects/p1/location/us-central1/deployments/my-deployment --gcs-source="gs://my-bucket"\n'
    git_source_repo_help = 'Repository URL.\nExample: \'https://github.com/examples/repository.git\'\n\nUse in conjunction with `--git-source-directory` and `--git-source_ref`\n\nExamples:\n\nCreate a deployment from the "https://github.com/examples/repository.git" repo, "staging/compute" folder, "mainline" branch:\n\n  $ {command} projects/p1/location/us-central1/deployments/my-deployment --git-source-repo="https://github.com/examples/repository.git"\n    --git-source-directory="staging/compute" --git-source-ref="mainline"\n'
    git_source_directory_help = 'Subdirectory inside the repository.\nExample: \'staging/my-package\'\n\nUse in conjunction with `--git-source-repo` and `--git-source-ref`\n\nExamples:\n\nCreate a deployment from the "https://github.com/examples/repository.git" repo, "staging/compute" folder, "mainline" branch:\n\n  $ {command} projects/p1/location/us-central1/deployments/my-deployment --git-source-repo="https://github.com/examples/repository.git"\n    --git-source-directory="staging/compute" --git-source-ref="mainline"\n'
    git_source_ref_help = 'Subdirectory inside the repository.\nExample: \'staging/my-package\'\n\nUse in conjunction with `--git-source-repo` and `--git-source-directory`\n\nExamples:\n\nCreate a deployment from the "https://github.com/examples/repository.git" repo, "staging/compute" folder, "mainline" branch:\n\n  $ {command} projects/p1/location/us-central1/deployments/my-deployment --git-source-repo="https://github.com/examples/repository.git"\n    --git-source-directory="staging/compute" --git-source-ref="mainline"\n'
    local_source_help = 'Local storage path where config files are stored. When using this option, terraform config file referecnes outside this storage path is not supported.\n      e.g. `./path/to/blueprint`\n\nExamples:\n\nCreate a deployment from a local storage path `./path/to/blueprint`:\n\n  $ {command} projects/p1/location/us-central1/deployments/my-deployment --local-source="./path/to/blueprint"\n'
    stage_bucket_help = 'Use in conjunction with `--local-source` to specify a destination storage bucket for\nuploading local files.\n\nIf unspecified, the bucket defaults to `gs://PROJECT_NAME_blueprints`. Uploaded\ncontent will appear in the `source` object under a name comprised of the\ntimestamp and a UUID. The final output destination looks like this:\n`gs://_BUCKET_/source/1615850562.234312-044e784992744951b0cd71c0b011edce/`\n\nExamples:\n\nCreate a deployment from a local storage path `./path/to/blueprint` and stage-bucket `gs://my-bucket`:\n\n  $ {command} projects/p1/location/us-central1/deployments/my-deployment --local-source="./path/to/blueprint" --stage-bucket="gs://my-bucket"\n'
    source_group = parser.add_group(mutex=False)
    input_values = source_group.add_mutually_exclusive_group()
    input_values.add_argument('--input-values', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help=input_values_help_text)
    input_values.add_argument('--inputs-file', help=inputs_file_help_text)
    source_details = source_group.add_mutually_exclusive_group()
    source_details.add_argument('--gcs-source', help=gcs_source_help_text)
    git_source_group = source_details.add_group(mutex=False)
    git_source_group.add_argument('--git-source-repo', help=git_source_repo_help)
    git_source_group.add_argument('--git-source-directory', help=git_source_directory_help)
    git_source_group.add_argument('--git-source-ref', help=git_source_ref_help)
    local_source_group = source_details.add_group(mutex=False)
    local_source_group.add_argument('--local-source', help=local_source_help)
    local_source_group.add_argument('--ignore-file', help='Override the `.gcloudignore` file and use the specified file instead. See `gcloud topic gcloudignore` for more information.')
    local_source_group.add_argument('--stage-bucket', help=stage_bucket_help, hidden=True, type=functions_api_util.ValidateAndStandarizeBucketUriOrRaise)