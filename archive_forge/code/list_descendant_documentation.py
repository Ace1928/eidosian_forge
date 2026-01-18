from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scc.manage.etd import clients
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc.manage import flags
from googlecloudsdk.command_lib.scc.manage import parsing
List the details of the resident and descendant Event Threat Detection custom modules.

  List the details of the resident and descendant Event Threat Detection
  custom modules for a specified organization or folder. For a project, this
  command lists only the custom modules that are created in the project.
  Modules created in a parent organization or folder are excluded from the
  list. To list the resident custom modules and the modules that are
  inherited from a parent organization and folder, use gcloud scc manage
  custom-modules etd list.

  ## EXAMPLES

  To list resident and descendant Event Threat Detection custom modules for
  organization `123`, run:

  $ {command} --organization=organizations/123

  To list resident and descendant Event Threat Detection custom modules for
  folder `456`, run:

  $ {command} --folder=folders/456

  To list resident and descendant Event Threat Detection custom modules for
  project `789`, run:

  $ {command} --project=projects/789
  