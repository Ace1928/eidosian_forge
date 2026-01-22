from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.github.prompt import (
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper
class GitHubToolkit(BaseToolkit):
    """GitHub Toolkit.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by creating, deleting, or updating,
        reading underlying data.

        For example, this toolkit can be used to create issues, pull requests,
        and comments on GitHub.

        See [Security](https://python.langchain.com/docs/security) for more information.
    """
    tools: List[BaseTool] = []

    @classmethod
    def from_github_api_wrapper(cls, github_api_wrapper: GitHubAPIWrapper) -> 'GitHubToolkit':
        operations: List[Dict] = [{'mode': 'get_issues', 'name': 'Get Issues', 'description': GET_ISSUES_PROMPT, 'args_schema': NoInput}, {'mode': 'get_issue', 'name': 'Get Issue', 'description': GET_ISSUE_PROMPT, 'args_schema': GetIssue}, {'mode': 'comment_on_issue', 'name': 'Comment on Issue', 'description': COMMENT_ON_ISSUE_PROMPT, 'args_schema': CommentOnIssue}, {'mode': 'list_open_pull_requests', 'name': 'List open pull requests (PRs)', 'description': LIST_PRS_PROMPT, 'args_schema': NoInput}, {'mode': 'get_pull_request', 'name': 'Get Pull Request', 'description': GET_PR_PROMPT, 'args_schema': GetPR}, {'mode': 'list_pull_request_files', 'name': 'Overview of files included in PR', 'description': LIST_PULL_REQUEST_FILES, 'args_schema': GetPR}, {'mode': 'create_pull_request', 'name': 'Create Pull Request', 'description': CREATE_PULL_REQUEST_PROMPT, 'args_schema': CreatePR}, {'mode': 'list_pull_request_files', 'name': "List Pull Requests' Files", 'description': LIST_PULL_REQUEST_FILES, 'args_schema': GetPR}, {'mode': 'create_file', 'name': 'Create File', 'description': CREATE_FILE_PROMPT, 'args_schema': CreateFile}, {'mode': 'read_file', 'name': 'Read File', 'description': READ_FILE_PROMPT, 'args_schema': ReadFile}, {'mode': 'update_file', 'name': 'Update File', 'description': UPDATE_FILE_PROMPT, 'args_schema': UpdateFile}, {'mode': 'delete_file', 'name': 'Delete File', 'description': DELETE_FILE_PROMPT, 'args_schema': DeleteFile}, {'mode': 'list_files_in_main_branch', 'name': 'Overview of existing files in Main branch', 'description': OVERVIEW_EXISTING_FILES_IN_MAIN, 'args_schema': NoInput}, {'mode': 'list_files_in_bot_branch', 'name': 'Overview of files in current working branch', 'description': OVERVIEW_EXISTING_FILES_BOT_BRANCH, 'args_schema': NoInput}, {'mode': 'list_branches_in_repo', 'name': 'List branches in this repository', 'description': LIST_BRANCHES_IN_REPO_PROMPT, 'args_schema': NoInput}, {'mode': 'set_active_branch', 'name': 'Set active branch', 'description': SET_ACTIVE_BRANCH_PROMPT, 'args_schema': BranchName}, {'mode': 'create_branch', 'name': 'Create a new branch', 'description': CREATE_BRANCH_PROMPT, 'args_schema': BranchName}, {'mode': 'get_files_from_directory', 'name': 'Get files from a directory', 'description': GET_FILES_FROM_DIRECTORY_PROMPT, 'args_schema': DirectoryPath}, {'mode': 'search_issues_and_prs', 'name': 'Search issues and pull requests', 'description': SEARCH_ISSUES_AND_PRS_PROMPT, 'args_schema': SearchIssuesAndPRs}, {'mode': 'search_code', 'name': 'Search code', 'description': SEARCH_CODE_PROMPT, 'args_schema': SearchCode}, {'mode': 'create_review_request', 'name': 'Create review request', 'description': CREATE_REVIEW_REQUEST_PROMPT, 'args_schema': CreateReviewRequest}]
        tools = [GitHubAction(name=action['name'], description=action['description'], mode=action['mode'], api_wrapper=github_api_wrapper, args_schema=action.get('args_schema', None)) for action in operations]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools