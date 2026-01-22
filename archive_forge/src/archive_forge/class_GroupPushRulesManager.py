from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupPushRulesManager(GetWithoutIdMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/groups/{group_id}/push_rule'
    _obj_cls = GroupPushRules
    _from_parent_attrs = {'group_id': 'id'}
    _create_attrs = RequiredOptional(optional=('deny_delete_tag', 'member_check', 'prevent_secrets', 'commit_message_regex', 'commit_message_negative_regex', 'branch_name_regex', 'author_email_regex', 'file_name_regex', 'max_file_size', 'commit_committer_check', 'reject_unsigned_commits'))
    _update_attrs = RequiredOptional(optional=('deny_delete_tag', 'member_check', 'prevent_secrets', 'commit_message_regex', 'commit_message_negative_regex', 'branch_name_regex', 'author_email_regex', 'file_name_regex', 'max_file_size', 'commit_committer_check', 'reject_unsigned_commits'))

    def get(self, **kwargs: Any) -> GroupPushRules:
        return cast(GroupPushRules, super().get(**kwargs))