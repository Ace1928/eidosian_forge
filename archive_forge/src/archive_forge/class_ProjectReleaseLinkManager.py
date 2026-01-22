from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import ArrayAttribute, RequiredOptional
class ProjectReleaseLinkManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/releases/{tag_name}/assets/links'
    _obj_cls = ProjectReleaseLink
    _from_parent_attrs = {'project_id': 'project_id', 'tag_name': 'tag_name'}
    _create_attrs = RequiredOptional(required=('name', 'url'), optional=('filepath', 'direct_asset_path', 'link_type'))
    _update_attrs = RequiredOptional(optional=('name', 'url', 'filepath', 'direct_asset_path', 'link_type'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectReleaseLink:
        return cast(ProjectReleaseLink, super().get(id=id, lazy=lazy, **kwargs))