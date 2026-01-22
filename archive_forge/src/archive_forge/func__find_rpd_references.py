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
def _find_rpd_references(context, stack_id):
    ev_ref_ids = set((e.rsrc_prop_data_id for e in context.session.query(models.Event).filter_by(stack_id=stack_id).all()))
    rsrc_ref_ids = set((r.rsrc_prop_data_id for r in context.session.query(models.Resource).filter_by(stack_id=stack_id).all()))
    return ev_ref_ids | rsrc_ref_ids