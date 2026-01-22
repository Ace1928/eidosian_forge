from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import iam_helpers
@staticmethod
def UpdateGsaIamPolicy(project_id, gsa_email, k8s_namespace, k8s_service_accounts):
    """Allow the k8s_service_accounts to use gsa_email via Workload Identity."""
    resource = 'projects/-/serviceAccounts/{gsa_email}'.format(gsa_email=gsa_email)
    members = ['serviceAccount:{project_id}.svc.id.goog[{k8s_namespace}/{ksa}]'.format(project_id=project_id, k8s_namespace=k8s_namespace, ksa=ksa) for ksa in k8s_service_accounts]
    iam_helpers.AddIamPolicyBindings(resource, members, 'roles/iam.workloadIdentityUser')