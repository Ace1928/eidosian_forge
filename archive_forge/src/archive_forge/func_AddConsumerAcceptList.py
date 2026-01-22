from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as forwarding_rule_flags
def AddConsumerAcceptList(parser):
    parser.add_argument('--consumer-accept-list', type=arg_parsers.ArgDict(), action='append', metavar='PROJECT_OR_NETWORK=LIMIT', default=None, help='    Specifies which consumer projects or networks are allowed to connect to the\n    service attachment. Each project or network has a connection limit. A given\n    service attachment can manage connections at either the project or network\n    level. Therefore, both the accept and reject lists for a given service\n    attachment must contain either only projects or only networks.\n\n    For example, `--consumer-accept-list myProjectId1=20` accepts a consumer\n    project myProjectId1 with connection limit 20;\n    `--consumer-accept-list projects/myProjectId1/global/networks/myNet1=20`\n    accepts a consumer network myNet1 with connection limit 20\n\n    * `PROJECT_OR_NETWORK` - Consumer project ID, project number or network URL.\n    * `CONNECTION_LIMIT` - The maximum number of allowed connections.\n    ')