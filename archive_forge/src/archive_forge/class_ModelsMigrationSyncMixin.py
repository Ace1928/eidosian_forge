import collections
import os
from alembic import command as alembic_command
from alembic import script as alembic_script
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import test_migrations
from sqlalchemy import sql
import sqlalchemy.types as types
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import versions
from glance.db.sqlalchemy import models
from glance.db.sqlalchemy import models_metadef
import glance.tests.utils as test_utils
class ModelsMigrationSyncMixin(object):

    def setUp(self):
        super(ModelsMigrationSyncMixin, self).setUp()
        self.engine = enginefacade.writer.get_engine()

    def get_metadata(self):
        for table in models_metadef.BASE_DICT.metadata.sorted_tables:
            models.BASE.metadata._add_table(table.name, table.schema, table)
        return models.BASE.metadata

    def get_engine(self):
        return self.engine

    def db_sync(self, engine):
        test_utils.db_sync(engine=engine)

    def compare_type(self, ctxt, insp_col, meta_col, insp_type, meta_type):
        if isinstance(meta_type, types.Variant):
            meta_orig_type = meta_col.type
            insp_orig_type = insp_col.type
            meta_col.type = meta_type.impl
            insp_col.type = meta_type.impl
            try:
                return self.compare_type(ctxt, insp_col, meta_col, insp_type, meta_type.impl)
            finally:
                meta_col.type = meta_orig_type
                insp_col.type = insp_orig_type
        else:
            ret = super(ModelsMigrationSyncMixin, self).compare_type(ctxt, insp_col, meta_col, insp_type, meta_type)
            if ret is not None:
                return ret
            return ctxt.impl.compare_type(insp_col, meta_col)

    def include_object(self, object_, name, type_, reflected, compare_to):
        if name in ['migrate_version'] and type_ == 'table':
            return False
        return True