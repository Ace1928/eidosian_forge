from sqlalchemy import inspect
from sqlalchemy import Integer
from ... import create_engine
from ... import exc
from ...schema import Column
from ...schema import DropConstraint
from ...schema import ForeignKeyConstraint
from ...schema import MetaData
from ...schema import Table
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import generate_driver_url
from ...testing.provision import get_temp_table_name
from ...testing.provision import log
from ...testing.provision import normalize_sequence
from ...testing.provision import run_reap_dbs
from ...testing.provision import temp_table_keyword_args
@temp_table_keyword_args.for_db('mssql')
def _mssql_temp_table_keyword_args(cfg, eng):
    return {}