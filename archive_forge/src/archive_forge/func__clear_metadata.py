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
def _clear_metadata(meta, conn):
    metadef_tables = [get_metadef_properties_table(meta, conn), get_metadef_objects_table(meta, conn), get_metadef_tags_table(meta, conn), get_metadef_namespace_resource_types_table(meta, conn), get_metadef_namespaces_table(meta, conn), get_metadef_resource_types_table(meta, conn)]
    with conn.begin():
        for table in metadef_tables:
            conn.execute(table.delete())
            LOG.info(_LI('Table %s has been cleared'), table)