from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class EnvironmentDeploymentBranchPolicyParams:
    """
    This class presents the deployment branch policy parameters as can be configured for an Environment.
    """

    def __init__(self, protected_branches: bool=False, custom_branch_policies: bool=False):
        assert isinstance(protected_branches, bool)
        assert isinstance(custom_branch_policies, bool)
        self.protected_branches = protected_branches
        self.custom_branch_policies = custom_branch_policies

    def _asdict(self) -> dict:
        return {'protected_branches': self.protected_branches, 'custom_branch_policies': self.custom_branch_policies}