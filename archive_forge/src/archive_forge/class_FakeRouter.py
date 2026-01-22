from unittest import mock
from oslo_config import cfg
from oslo_db.sqlalchemy import models
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.api import attributes
from neutron_lib import context
from neutron_lib.db import utils
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
class FakeRouter(ModelBaseV2):
    __tablename__ = 'fakerouters'
    router_id = sa.Column(sa.String(36), primary_key=True)
    gw_port_id = sa.Column(sa.String(36), sa.ForeignKey(FakePort.port_id))
    gw_port = orm.relationship(FakePort, lazy='joined')