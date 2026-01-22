from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import service_utils
from heat.db import api as db_api
from heat.objects import base as heat_base
@classmethod
def _from_db_objects(cls, context, list_obj):
    return [cls._from_db_object(context, cls(context), obj) for obj in list_obj]