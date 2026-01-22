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
def _update_flow_details(self, conn, fd, e_fd):
    e_fd.merge(fd)
    conn.execute(sql.update(self._tables.flowdetails).where(self._tables.flowdetails.c.uuid == e_fd.uuid).values(e_fd.to_dict()))
    for ad in fd:
        e_ad = e_fd.find(ad.uuid)
        if e_ad is None:
            e_fd.add(ad)
            self._insert_atom_details(conn, ad, fd.uuid)
        else:
            self._update_atom_details(conn, ad, e_ad)