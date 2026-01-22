from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class GroupAccessRequestManager(ListMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/groups/{group_id}/access_requests'
    _obj_cls = GroupAccessRequest
    _from_parent_attrs = {'group_id': 'id'}