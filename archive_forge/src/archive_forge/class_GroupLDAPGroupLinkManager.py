from typing import Any, BinaryIO, cast, Dict, List, Optional, Type, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .access_requests import GroupAccessRequestManager  # noqa: F401
from .audit_events import GroupAuditEventManager  # noqa: F401
from .badges import GroupBadgeManager  # noqa: F401
from .boards import GroupBoardManager  # noqa: F401
from .clusters import GroupClusterManager  # noqa: F401
from .container_registry import GroupRegistryRepositoryManager  # noqa: F401
from .custom_attributes import GroupCustomAttributeManager  # noqa: F401
from .deploy_tokens import GroupDeployTokenManager  # noqa: F401
from .epics import GroupEpicManager  # noqa: F401
from .export_import import GroupExportManager, GroupImportManager  # noqa: F401
from .group_access_tokens import GroupAccessTokenManager  # noqa: F401
from .hooks import GroupHookManager  # noqa: F401
from .invitations import GroupInvitationManager  # noqa: F401
from .issues import GroupIssueManager  # noqa: F401
from .iterations import GroupIterationManager  # noqa: F401
from .labels import GroupLabelManager  # noqa: F401
from .members import (  # noqa: F401
from .merge_requests import GroupMergeRequestManager  # noqa: F401
from .milestones import GroupMilestoneManager  # noqa: F401
from .notification_settings import GroupNotificationSettingsManager  # noqa: F401
from .packages import GroupPackageManager  # noqa: F401
from .projects import GroupProjectManager, SharedProjectManager  # noqa: F401
from .push_rules import GroupPushRulesManager
from .runners import GroupRunnerManager  # noqa: F401
from .statistics import GroupIssuesStatisticsManager  # noqa: F401
from .variables import GroupVariableManager  # noqa: F401
from .wikis import GroupWikiManager  # noqa: F401
class GroupLDAPGroupLinkManager(ListMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/groups/{group_id}/ldap_group_links'
    _obj_cls: Type[GroupLDAPGroupLink] = GroupLDAPGroupLink
    _from_parent_attrs = {'group_id': 'id'}
    _create_attrs = RequiredOptional(required=('provider', 'group_access'), exclusive=('cn', 'filter'))