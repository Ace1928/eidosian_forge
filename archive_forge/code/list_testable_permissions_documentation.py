from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.iam import exceptions
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import flags
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import resources
List IAM testable permissions for a resource.

  Testable permissions mean the permissions that user can add or remove in
  a role at a given resource.
  The resource can be referenced either via the full resource name or via a URI.

  ## EXAMPLES

  List testable permissions for a resource identified via full resource name:

    $ {command} //cloudresourcemanager.googleapis.com/organizations/1234567

  List testable permissions for a resource identified via URI:

    $ {command} https://www.googleapis.com/compute/v1/projects/example-project
  