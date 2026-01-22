from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from typing import Optional
from absl import app
from absl import flags
import bq_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from utils import bq_id_utils
class SetIamPolicy(_IamPolicyCmd):
    usage = 'set-iam-policy [(-d|-t|-connection)] <identifier> <filename>'

    def __init__(self, name: str, fv: flags.FlagValues):
        super(SetIamPolicy, self).__init__(name, fv, 'Set')
        self._ProcessCommandRc(fv)

    def RunWithArgs(self, identifier: str, filename: str) -> Optional[int]:
        """Set the IAM policy for a resource.

    Sets the IAM policy for a dataset, table or connection resource. After
    setting the policy, the new policy is printed to stdout. Policies are in
    JSON format.

    If the 'etag' field is present in the policy, it must match the value in the
    current policy, which can be obtained with 'bq get-iam-policy'. Otherwise
    this command will fail. This feature allows users to prevent concurrent
    updates.

    Usage:
    set-iam-policy <identifier> <filename>

    Examples:
      bq set-iam-policy ds.table1 /tmp/policy.json
      bq set-iam-policy --project_id=proj -t ds.table1 /tmp/policy.json
      bq set-iam-policy proj:ds.table1 /tmp/policy.json

    Arguments:
      identifier: The identifier of the resource. Presently only table, view and
        connection resources are fully supported. (Last updated: 2022-04-25)
      filename: The name of a file containing the policy in JSON format.
    """
        client = bq_cached_client.Client.Get()
        reference = self.GetReferenceFromIdentifier(client, identifier)
        with open(filename, 'r') as file_obj:
            policy = json.load(file_obj)
            result_policy = self.SetPolicyForReference(client, reference, policy)
            bq_utils.PrintFormattedJsonObject(result_policy, default_format='prettyjson')