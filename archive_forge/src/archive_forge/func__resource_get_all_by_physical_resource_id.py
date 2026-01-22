import datetime
import functools
import itertools
import random
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import orm
from heat.common import crypt
from heat.common import exception
from heat.common.i18n import _
from heat.db import filters as db_filters
from heat.db import models
from heat.db import utils as db_utils
from heat.engine import environment as heat_environment
from heat.rpc import api as rpc_api
def _resource_get_all_by_physical_resource_id(context, physical_resource_id):
    results = context.session.query(models.Resource).filter_by(physical_resource_id=physical_resource_id).options(orm.joinedload(models.Resource.attr_data), orm.joinedload(models.Resource.data), orm.joinedload(models.Resource.rsrc_prop_data)).all()
    for result in results:
        if context is None or context.is_admin or context.tenant_id in (result.stack.tenant, result.stack.stack_user_project_id):
            yield result