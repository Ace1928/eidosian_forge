import copy
from oslo_config import cfg
from oslo_log import log as logging
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import crypt
from heat.common import environment_format as env_fmt
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
RawTemplate object.