import json
import os
from os.path import isfile
from os.path import join
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy.schema import MetaData
from sqlalchemy.sql import select
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
def _get_resource_id(table, conn, namespace_id, resource_name):
    with conn.begin():
        resource = conn.execute(select(table.c.id).where(and_(table.c.namespace_id == namespace_id, table.c.name == resource_name)).select_from(table)).fetchone()
        if resource:
            return resource[0]
    return None