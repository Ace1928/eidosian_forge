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
class GetIamPolicy(_IamPolicyCmd):
    usage = 'get-iam-policy [(-d|-t|-connection)] <identifier>'

    def __init__(self, name: str, fv: flags.FlagValues):
        super(GetIamPolicy, self).__init__(name, fv, 'Get')
        self._ProcessCommandRc(fv)

    def RunWithArgs(self, identifier: str) -> Optional[int]:
        """Get the IAM policy for a resource.

    Gets the IAM policy for a dataset, table or connection resource, and prints
    it to stdout. The policy is in JSON format.

    Usage:
    get-iam-policy <identifier>

    Examples:
      bq get-iam-policy ds.table1
      bq get-iam-policy --project_id=proj -t ds.table1
      bq get-iam-policy proj:ds.table1

    Arguments:
      identifier: The identifier of the resource. Presently only table, view and
        connection resources are fully supported. (Last updated: 2022-04-25)
    """
        client = bq_cached_client.Client.Get()
        reference = self.GetReferenceFromIdentifier(client, identifier)
        result_policy = self.GetPolicyForReference(client, reference)
        bq_utils.PrintFormattedJsonObject(result_policy, default_format='prettyjson')