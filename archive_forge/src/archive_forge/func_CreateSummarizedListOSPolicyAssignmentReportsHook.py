from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def CreateSummarizedListOSPolicyAssignmentReportsHook(response, args):
    """Create ListTableRow from ListOSPolicyAssignmentReports response.

  Args:
    response: Response from ListOSPolicyAssignmentReports
    args: gcloud args

  Returns:
    ListTableRows of summarized os policy assignment reports
  """
    rows = []
    for report in response:
        compliant_policies_count = 0
        total_policies_count = 0
        for policy in report.osPolicyCompliances:
            total_policies_count += 1
            if policy.complianceState.name == 'COMPLIANT':
                compliant_policies_count += 1
        summary_str = _LIST_SUMMARY_STR.format(compliant_policies_count=compliant_policies_count, total_policies_count=total_policies_count)
        rows.append(ListTableRow(instance=report.instance, assignment_id=report.name.split('/')[-2], location=args.location or properties.VALUES.compute.zone.Get(), update_time=report.updateTime, summary_str=summary_str))
    return rows