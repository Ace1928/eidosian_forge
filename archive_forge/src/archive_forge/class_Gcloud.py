from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import properties
class Gcloud(base.Group):
    """Manage Google Cloud resources and developer workflow.

  The `gcloud` CLI manages authentication, local configuration, developer
  workflow, and interactions with the Google Cloud APIs.

  For a quick introduction to the `gcloud` CLI, a list of commonly
  used commands, and a look at how these commands are structured, run
  `gcloud cheat-sheet` or see the
  [`gcloud` CLI cheat sheet](https://cloud.google.com/sdk/docs/cheatsheet).
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('--account', metavar='ACCOUNT', category=base.COMMONLY_USED_FLAGS, help='Google Cloud user account to use for invocation.', action=actions.StoreProperty(properties.VALUES.core.account))
        parser.add_argument('--universe-domain', metavar='UNIVERSE_DOMAIN', category=base.COMMONLY_USED_FLAGS, help='Universe domain to target.', hidden=True, action=actions.StoreProperty(properties.VALUES.core.universe_domain))
        parser.add_argument('--impersonate-service-account', metavar='SERVICE_ACCOUNT_EMAILS', help="        For this `gcloud` invocation, all API requests will be\n        made as the given service account or target service account in an\n        impersonation delegation chain instead of the currently selected\n        account. You can specify either a single service account as the\n        impersonator, or a comma-separated list of service accounts to\n        create an impersonation delegation chain. The impersonation is done\n        without needing to create, download, and activate a key for the\n        service account or accounts.\n\n        In order to make API requests as a service account, your\n        currently selected account must have an IAM role that includes\n        the `iam.serviceAccounts.getAccessToken` permission for the\n        service account or accounts.\n\n        The `roles/iam.serviceAccountTokenCreator` role has\n        the `iam.serviceAccounts.getAccessToken permission`. You can\n        also create a custom role.\n\n        You can specify a list of service accounts, separated with\n        commas. This creates an impersonation delegation chain in which\n        each service account delegates its permissions to the next\n        service account in the chain. Each service account in the list\n        must have the `roles/iam.serviceAccountTokenCreator` role on the\n        next service account in the list. For example, when\n        `--impersonate-service-account=`\n        ``SERVICE_ACCOUNT_1'',``SERVICE_ACCOUNT_2'',\n        the active account must have the\n        `roles/iam.serviceAccountTokenCreator` role on\n        ``SERVICE_ACCOUNT_1'', which must have the\n        `roles/iam.serviceAccountTokenCreator` role on\n        ``SERVICE_ACCOUNT_2''.\n        ``SERVICE_ACCOUNT_1'' is the impersonated service\n        account and ``SERVICE_ACCOUNT_2'' is the delegate.\n        ", action=actions.StoreProperty(properties.VALUES.auth.impersonate_service_account))
        parser.add_argument('--access-token-file', metavar='ACCESS_TOKEN_FILE', help='        A file path to read the access token. Use this flag to\n        authenticate `gcloud` with an access token. The credentials of\n        the active account (if exists) will be ignored. The file should\n        only contain an access token with no other information.\n        ', action=actions.StoreProperty(properties.VALUES.auth.access_token_file))
        common_args.ProjectArgument().AddToParser(parser)
        parser.add_argument('--billing-project', metavar='BILLING_PROJECT', category=base.COMMONLY_USED_FLAGS, help='             The Google Cloud project that will be charged quota for\n             operations performed in `gcloud`. If you need to operate on one\n             project, but need quota against a different project, you can use\n             this flag to specify the billing project. If both\n             `billing/quota_project` and `--billing-project` are specified,\n             `--billing-project` takes precedence.\n             Run `$ gcloud config set --help` to see more information about\n             `billing/quota_project`.\n             ', action=actions.StoreProperty(properties.VALUES.billing.quota_project))
        parser.add_argument('--quiet', '-q', default=None, category=base.COMMONLY_USED_FLAGS, action=actions.StoreConstProperty(properties.VALUES.core.disable_prompts, True), help='        Disable all interactive prompts when running `gcloud` commands. If input\n        is required, defaults will be used, or an error will be raised.\n\n        Overrides the default core/disable_prompts property value for this\n        command invocation. This is equivalent to setting the environment\n        variable `CLOUDSDK_CORE_DISABLE_PROMPTS` to 1.\n        ')
        trace_group = parser.add_mutually_exclusive_group()
        trace_group.add_argument('--trace-token', default=None, action=actions.StoreProperty(properties.VALUES.core.trace_token), help='Token used to route traces of service requests for investigation of issues.')