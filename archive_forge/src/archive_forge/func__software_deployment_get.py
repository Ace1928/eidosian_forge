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
def _software_deployment_get(context, deployment_id):
    result = context.session.query(models.SoftwareDeployment).filter_by(id=deployment_id).options(orm.joinedload(models.SoftwareDeployment.config)).first()
    if result is not None and context is not None and (not context.is_admin) and (context.tenant_id not in (result.tenant, result.stack_user_project_id)):
        result = None
    if not result:
        raise exception.NotFound(_('Deployment with id %s not found') % deployment_id)
    return result