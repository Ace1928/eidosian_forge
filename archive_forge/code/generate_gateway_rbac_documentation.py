from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.fleet import rbac_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
Generate RBAC policy files for connected clusters by the user.

  {command} generates RBAC policies to be used by Connect Gateway API.

  Upon success, this command will write the output RBAC policy to the designated
  local file in dry run mode.

  Override RBAC policy: Y to override previous RBAC policy, N to stop. If
  overriding the --role, Y will clean up the previous RBAC policy and then apply
  the new one.

  ## EXAMPLES
    The current implementation supports multiple modes:

    Dry run mode to generate the RBAC policy file, and write to local directory:

      $ {command} --membership=my-cluster
      --users=foo@example.com,test-acct@test-project.iam.gserviceaccount.com
      --role=clusterrole/cluster-admin --rbac-output-file=./rbac.yaml

    Dry run mode to generate the RBAC policy, and print on screen:

      $ {command} --membership=my-cluster
      --users=foo@example.com,test-acct@test-project.iam.gserviceaccount.com
      --role=clusterrole/cluster-admin

    Anthos support mode, generate the RBAC policy file with read-only permission
    for TSE/Eng to debug customers' clusters:

      $ {command} --membership=my-cluster --anthos-support

    Apply mode, generate the RBAC policy and apply it to the specified cluster:

      $ {command} --membership=my-cluster
      --users=foo@example.com,test-acct@test-project.iam.gserviceaccount.com
      --role=clusterrole/cluster-admin --context=my-cluster-context
      --kubeconfig=/home/user/custom_kubeconfig --apply

    Revoke mode, revoke the RBAC policy for the specified users:

      $ {command} --membership=my-cluster
      --users=foo@example.com,test-acct@test-project.iam.gserviceaccount.com
      --role=clusterrole/cluster-admin --context=my-cluster-context
      --kubeconfig=/home/user/custom_kubeconfig --revoke

    The role to be granted to the users can either be cluster-scoped or
    namespace-scoped. To grant a namespace-scoped role to the users in dry run
    mode, run:

      $ {command} --membership=my-cluster
      --users=foo@example.com,test-acct@test-project.iam.gserviceaccount.com
      --role=role/mynamespace/namespace-reader

    The users provided can be using a Google identity (only email) or using
    external identity providers (starting with
    "principal://iam.googleapis.com"):

      $ {command} --membership=my-cluster
      --users=foo@example.com,principal://iam.googleapis.com/locations/global/workforcePools/pool/subject/user
      --role=clusterrole/cluster-admin --context=my-cluster-context
      --kubeconfig=/home/user/custom_kubeconfig --apply

    The groups can be provided as a Google identity (only email) or an
    external identity (starting with
    "principalSet://iam.googleapis.com"):

      $ {command} --membership=my-cluster
      --groups=group@example.com,principalSet://iam.googleapis.com/locations/global/workforcePools/pool/group/ExampleGroup
      --role=clusterrole/cluster-admin --context=my-cluster-context
      --kubeconfig=/home/user/custom_kubeconfig --apply
  