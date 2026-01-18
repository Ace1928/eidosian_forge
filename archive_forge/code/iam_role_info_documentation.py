from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_instance_profiles
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_role_attached_policies
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_role_policies
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_roles
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry

    Module action handler
    