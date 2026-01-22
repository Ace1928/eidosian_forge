from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin, RetrieveMixin
class ProjectEventManager(EventManager):
    _path = '/projects/{project_id}/events'
    _obj_cls = ProjectEvent
    _from_parent_attrs = {'project_id': 'id'}