from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetAnalyzeIamPolicyRequest(_messages.Message):
    """A CloudassetAnalyzeIamPolicyRequest object.

  Fields:
    analysisQuery_accessSelector_permissions: Optional. The permissions to
      appear in result.
    analysisQuery_accessSelector_roles: Optional. The roles to appear in
      result.
    analysisQuery_conditionContext_accessTime: The hypothetical access
      timestamp to evaluate IAM conditions. Note that this value must not be
      earlier than the current time; otherwise, an INVALID_ARGUMENT error will
      be returned.
    analysisQuery_identitySelector_identity: Required. The identity appear in
      the form of principals in [IAM policy
      binding](https://cloud.google.com/iam/reference/rest/v1/Binding). The
      examples of supported forms are: "user:mike@example.com",
      "group:admins@example.com", "domain:google.com", "serviceAccount:my-
      project-id@appspot.gserviceaccount.com". Notice that wildcard characters
      (such as * and ?) are not supported. You must give a specific identity.
    analysisQuery_options_analyzeServiceAccountImpersonation: Optional. If
      true, the response will include access analysis from identities to
      resources via service account impersonation. This is a very expensive
      operation, because many derived queries will be executed. We highly
      recommend you use AssetService.AnalyzeIamPolicyLongrunning RPC instead.
      For example, if the request analyzes for which resources user A has
      permission P, and there's an IAM policy states user A has
      iam.serviceAccounts.getAccessToken permission to a service account SA,
      and there's another IAM policy states service account SA has permission
      P to a Google Cloud folder F, then user A potentially has access to the
      Google Cloud folder F. And those advanced analysis results will be
      included in
      AnalyzeIamPolicyResponse.service_account_impersonation_analysis. Another
      example, if the request analyzes for who has permission P to a Google
      Cloud folder F, and there's an IAM policy states user A has
      iam.serviceAccounts.actAs permission to a service account SA, and
      there's another IAM policy states service account SA has permission P to
      the Google Cloud folder F, then user A potentially has access to the
      Google Cloud folder F. And those advanced analysis results will be
      included in
      AnalyzeIamPolicyResponse.service_account_impersonation_analysis. Only
      the following permissions are considered in this analysis: *
      `iam.serviceAccounts.actAs` * `iam.serviceAccounts.signBlob` *
      `iam.serviceAccounts.signJwt` * `iam.serviceAccounts.getAccessToken` *
      `iam.serviceAccounts.getOpenIdToken` *
      `iam.serviceAccounts.implicitDelegation` Default is false.
    analysisQuery_options_expandGroups: Optional. If true, the identities
      section of the result will expand any Google groups appearing in an IAM
      policy binding. If IamPolicyAnalysisQuery.identity_selector is
      specified, the identity in the result will be determined by the
      selector, and this flag is not allowed to set. If true, the default max
      expansion per group is 1000 for AssetService.AnalyzeIamPolicy][].
      Default is false.
    analysisQuery_options_expandResources: Optional. If true and
      IamPolicyAnalysisQuery.resource_selector is not specified, the resource
      section of the result will expand any resource attached to an IAM policy
      to include resources lower in the resource hierarchy. For example, if
      the request analyzes for which resources user A has permission P, and
      the results include an IAM policy with P on a Google Cloud folder, the
      results will also include resources in that folder with permission P. If
      true and IamPolicyAnalysisQuery.resource_selector is specified, the
      resource section of the result will expand the specified resource to
      include resources lower in the resource hierarchy. Only project or lower
      resources are supported. Folder and organization resources cannot be
      used together with this option. For example, if the request analyzes for
      which users have permission P on a Google Cloud project with this option
      enabled, the results will include all users who have permission P on
      that project or any lower resource. If true, the default max expansion
      per resource is 1000 for AssetService.AnalyzeIamPolicy][] and 100000 for
      AssetService.AnalyzeIamPolicyLongrunning][]. Default is false.
    analysisQuery_options_expandRoles: Optional. If true, the access section
      of result will expand any roles appearing in IAM policy bindings to
      include their permissions. If IamPolicyAnalysisQuery.access_selector is
      specified, the access section of the result will be determined by the
      selector, and this flag is not allowed to set. Default is false.
    analysisQuery_options_includeDenyPolicyAnalysis: Optional. If true, the
      response includes deny policy analysis results for access tuples. The
      deny policy analysis will be conducted on max 1000 access tuples. For
      access tuples don't have deny policy analysis result populated, you can
      issue another query of that access tuple to get deny policy analysis
      result for it. Default is false.
    analysisQuery_options_outputGroupEdges: Optional. If true, the result will
      output the relevant membership relationships between groups and other
      groups, and between groups and principals. Default is false.
    analysisQuery_options_outputResourceEdges: Optional. If true, the result
      will output the relevant parent/child relationships between resources.
      Default is false.
    analysisQuery_resourceSelector_fullResourceName: Required. The [full
      resource name] (https://cloud.google.com/asset-inventory/docs/resource-
      name-format) of a resource of [supported resource
      types](https://cloud.google.com/asset-inventory/docs/supported-asset-
      types#analyzable_asset_types).
    executionTimeout: Optional. Amount of time executable has to complete. See
      JSON representation of
      [Duration](https://developers.google.com/protocol-
      buffers/docs/proto3#json). If this field is set with a value less than
      the RPC deadline, and the execution of your query hasn't finished in the
      specified execution timeout, you will get a response with partial
      result. Otherwise, your query's execution will continue until the RPC
      deadline. If it's not finished until then, you will get a
      DEADLINE_EXCEEDED error. Default is empty.
    savedAnalysisQuery: Optional. The name of a saved query, which must be in
      the format of: * projects/project_number/savedQueries/saved_query_id *
      folders/folder_number/savedQueries/saved_query_id *
      organizations/organization_number/savedQueries/saved_query_id If both
      `analysis_query` and `saved_analysis_query` are provided, they will be
      merged together with the `saved_analysis_query` as base and the
      `analysis_query` as overrides. For more details of the merge behavior,
      refer to the [MergeFrom](https://developers.google.com/protocol-buffers/
      docs/reference/cpp/google.protobuf.message#Message.MergeFrom.details)
      page. Note that you cannot override primitive fields with default value,
      such as 0 or empty string, etc., because we use proto3, which doesn't
      support field presence yet.
    scope: Required. The relative name of the root asset. Only resources and
      IAM policies within the scope will be analyzed. This can only be an
      organization number (such as "organizations/123"), a folder number (such
      as "folders/123"), a project ID (such as "projects/my-project-id"), or a
      project number (such as "projects/12345"). To know how to get
      organization ID, visit [here ](https://cloud.google.com/resource-
      manager/docs/creating-managing-
      organization#retrieving_your_organization_id). To know how to get folder
      or project ID, visit [here ](https://cloud.google.com/resource-
      manager/docs/creating-managing-
      folders#viewing_or_listing_folders_and_projects).
  """
    analysisQuery_accessSelector_permissions = _messages.StringField(1, repeated=True)
    analysisQuery_accessSelector_roles = _messages.StringField(2, repeated=True)
    analysisQuery_conditionContext_accessTime = _messages.StringField(3)
    analysisQuery_identitySelector_identity = _messages.StringField(4)
    analysisQuery_options_analyzeServiceAccountImpersonation = _messages.BooleanField(5)
    analysisQuery_options_expandGroups = _messages.BooleanField(6)
    analysisQuery_options_expandResources = _messages.BooleanField(7)
    analysisQuery_options_expandRoles = _messages.BooleanField(8)
    analysisQuery_options_includeDenyPolicyAnalysis = _messages.BooleanField(9)
    analysisQuery_options_outputGroupEdges = _messages.BooleanField(10)
    analysisQuery_options_outputResourceEdges = _messages.BooleanField(11)
    analysisQuery_resourceSelector_fullResourceName = _messages.StringField(12)
    executionTimeout = _messages.StringField(13)
    savedAnalysisQuery = _messages.StringField(14)
    scope = _messages.StringField(15, required=True)