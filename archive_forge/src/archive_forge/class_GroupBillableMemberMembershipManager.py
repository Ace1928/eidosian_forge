from typing import Any, cast, Union
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupBillableMemberMembershipManager(ListMixin, RESTManager):
    _path = '/groups/{group_id}/billable_members/{user_id}/memberships'
    _obj_cls = GroupBillableMemberMembership
    _from_parent_attrs = {'group_id': 'group_id', 'user_id': 'id'}