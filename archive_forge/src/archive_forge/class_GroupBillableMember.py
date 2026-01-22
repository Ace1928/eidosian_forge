from typing import Any, cast, Union
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupBillableMember(ObjectDeleteMixin, RESTObject):
    _repr_attr = 'username'
    memberships: 'GroupBillableMemberMembershipManager'