from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Set
from sqlalchemy import CHAR
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Numeric
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from ... import autogenerate
from ... import util
from ...autogenerate import api
from ...ddl.base import _fk_spec
from ...migration import MigrationContext
from ...operations import ops
from ...testing import config
from ...testing import eq_
from ...testing.env import clear_staging_env
from ...testing.env import staging_env
class AutogenTest(_ComparesFKs):

    def _flatten_diffs(self, diffs):
        for d in diffs:
            if isinstance(d, list):
                yield from self._flatten_diffs(d)
            else:
                yield d

    @classmethod
    def _get_bind(cls):
        return config.db
    configure_opts: Dict[Any, Any] = {}

    @classmethod
    def setup_class(cls):
        staging_env()
        cls.bind = cls._get_bind()
        cls.m1 = cls._get_db_schema()
        cls.m1.create_all(cls.bind)
        cls.m2 = cls._get_model_schema()

    @classmethod
    def teardown_class(cls):
        cls.m1.drop_all(cls.bind)
        clear_staging_env()

    def setUp(self):
        self.conn = conn = self.bind.connect()
        ctx_opts = {'compare_type': True, 'compare_server_default': True, 'target_metadata': self.m2, 'upgrade_token': 'upgrades', 'downgrade_token': 'downgrades', 'alembic_module_prefix': 'op.', 'sqlalchemy_module_prefix': 'sa.', 'include_object': _default_object_filters, 'include_name': _default_name_filters}
        if self.configure_opts:
            ctx_opts.update(self.configure_opts)
        self.context = context = MigrationContext.configure(connection=conn, opts=ctx_opts)
        self.autogen_context = api.AutogenContext(context, self.m2)

    def tearDown(self):
        self.conn.close()

    def _update_context(self, object_filters=None, name_filters=None, include_schemas=None):
        if include_schemas is not None:
            self.autogen_context.opts['include_schemas'] = include_schemas
        if object_filters is not None:
            self.autogen_context._object_filters = [object_filters]
        if name_filters is not None:
            self.autogen_context._name_filters = [name_filters]
        return self.autogen_context