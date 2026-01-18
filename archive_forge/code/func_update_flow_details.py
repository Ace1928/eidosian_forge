import copy
import functools
import threading
import time
from oslo_utils import strutils
import sqlalchemy as sa
from sqlalchemy import exc as sa_exc
from sqlalchemy import pool as sa_pool
from sqlalchemy import sql
import tenacity
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence.backends.sqlalchemy import migration
from taskflow.persistence.backends.sqlalchemy import tables
from taskflow.persistence import base
from taskflow.persistence import models
from taskflow.utils import eventlet_utils
from taskflow.utils import misc
def update_flow_details(self, flow_detail):
    try:
        flowdetails = self._tables.flowdetails
        with self._engine.begin() as conn:
            q = sql.select(flowdetails).where(flowdetails.c.uuid == flow_detail.uuid)
            row = conn.execute(q).first()
            if not row:
                raise exc.NotFound("No flow details found with uuid '%s'" % flow_detail.uuid)
            row = row._mapping
            e_fd = self._converter.convert_flow_detail(row)
            self._converter.populate_flow_detail(conn, e_fd)
            self._update_flow_details(conn, flow_detail, e_fd)
        return e_fd
    except sa_exc.SQLAlchemyError:
        exc.raise_with_cause(exc.StorageFailure, "Failed updating flow details with uuid '%s'" % flow_detail.uuid)