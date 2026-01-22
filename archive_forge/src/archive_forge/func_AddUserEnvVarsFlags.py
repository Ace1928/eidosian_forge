from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def AddUserEnvVarsFlags(parser):
    """Adds flags for configuring user-defined environment variables."""
    userenvvars_group = parser.add_group(mutex=True, hidden=True)
    userenvvars_group.add_argument('--set-env-vars', type=arg_parsers.ArgDict(key_type=str, value_type=str, max_length=USER_ENV_VARS_LIMIT), action=arg_parsers.UpdateAction, metavar='KEY=VALUE', help='        Sets customer-defined environment variables used in the new workflow\n        revision.\n\n        This flag takes a comma-separated list of key value pairs.\n        Example:\n        gcloud workflows deploy ${workflow_name} --set-env-vars foo=bar,hey=hi...\n      ')
    map_util.AddMapSetFileFlag(userenvvars_group, 'env-vars', 'environment variables', key_type=str, value_type=str)
    userenvvars_group.add_argument('--clear-env-vars', action='store_true', help='        Clears customer-defined environment variables used in the new workflow\n        revision.\n\n        Example:\n        gcloud workflows deploy ${workflow_name} --clear-env-vars\n      ')
    userenvvars_group.add_argument('--remove-env-vars', metavar='KEY', action=arg_parsers.UpdateAction, type=arg_parsers.ArgList(element_type=str), help='        Removes customer-defined environment variables used in the new workflow\n        revision.\n        It takes a list of environment variables keys to be removed.\n\n        Example:\n        gcloud workflows deploy ${workflow_name} --remove-env-vars foo,hey...\n      ')
    userenvvars_group.add_argument('--update-env-vars', type=arg_parsers.ArgDict(key_type=str, value_type=str), action=arg_parsers.UpdateAction, metavar='KEY=VALUE', help='        Updates existing or adds new customer-defined environment variables used\n        in the new workflow revision.\n\n        This flag takes a comma-separated list of key value pairs.\n        Example:\n        gcloud workflows deploy ${workflow_name} --update-env-vars foo=bar,hey=hi\n      ')