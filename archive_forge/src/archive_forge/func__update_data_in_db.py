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
def _update_data_in_db(conn, table, values, column, value):
    try:
        with conn.begin():
            conn.execute(table.update().values(values).where(column == value))
    except sqlalchemy.exc.IntegrityError:
        LOG.warning(_LW('Duplicate entry for values: %s'), values)