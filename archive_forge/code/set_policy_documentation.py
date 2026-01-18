from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import org_policies_base
from googlecloudsdk.command_lib.resource_manager import org_policies_flags as flags
Set Organization Policy.

  Sets an Organization Policy associated with the specified resource from
  a file that contains the JSON or YAML encoded Organization Policy.

  ## EXAMPLES

  Organization policy list constraint YAML file example:

    constraint: constraints/CONSTRAINT_NAME
    listPolicy:
      deniedValues:
      - VALUE_A

  Organization policy list constraint JSON file example:

    {
      "constraint": "constraints/CONSTRAINT_NAME",
      "listPolicy": {
        "deniedValues": ["VALUE_A"]
      }
    }

  The following command sets an Organization Policy for a constraint
  on project `foo-project` from file `/tmp/policy.yaml`:

    $ {command} --project=foo-project /tmp/policy.yaml
  