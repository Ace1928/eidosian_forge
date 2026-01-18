from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scc.manage.sha import clients
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc.manage import flags
from googlecloudsdk.command_lib.scc.manage import parsing
Command to simulate a SHA custom module.

  ## EXAMPLES

  To simulate a Security Health Analytics custom module with
  ID `123456` for organization `123`, run:

    $ {command} 123456
    --organization=123
    --custom-config-from-file=custom_config.yaml
    --resource-from-file=test.yaml

  To simulate a Security Health Analytics custom module with
  ID `123456` for folder `456`, run:

    $ {command} 123456
    --folder=456
    --custom-config-from-file=custom_config.yaml
    --resource-from-file=test.yaml

  To simulate a Security Health Analytics custom module with
  ID `123456` for project `789`, run:

    $ {command} 123456
    --project=789
    --custom-config-from-file=custom_config.yaml
    --resource-from-file=test.yaml

  You can also specify the parent more generally:

    $ {command} 123456
    --parent=organizations/123
    --custom-config-from-file=custom_config.yaml
    --resource-from-file=test.yaml

  Or just specify the fully qualified module name:

    $ {command}
    organizations/123/locations/global/effectiveSecurityHealthAnalyticsCustomModules/123456
    --custom-config-from-file=custom_config.yaml
    --resource-from-file=test.yaml
  