from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, NoUpdateMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class GroupHookManager(CRUDMixin, RESTManager):
    _path = '/groups/{group_id}/hooks'
    _obj_cls = GroupHook
    _from_parent_attrs = {'group_id': 'id'}
    _create_attrs = RequiredOptional(required=('url',), optional=('push_events', 'issues_events', 'confidential_issues_events', 'merge_requests_events', 'tag_push_events', 'note_events', 'confidential_note_events', 'job_events', 'pipeline_events', 'wiki_page_events', 'deployment_events', 'releases_events', 'subgroup_events', 'enable_ssl_verification', 'token'))
    _update_attrs = RequiredOptional(required=('url',), optional=('push_events', 'issues_events', 'confidential_issues_events', 'merge_requests_events', 'tag_push_events', 'note_events', 'confidential_note_events', 'job_events', 'pipeline_events', 'wiki_page_events', 'deployment_events', 'releases_events', 'subgroup_events', 'enable_ssl_verification', 'token'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupHook:
        return cast(GroupHook, super().get(id=id, lazy=lazy, **kwargs))