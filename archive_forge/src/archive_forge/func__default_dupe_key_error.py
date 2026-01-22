import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('mysql', sqla_exc.IntegrityError, "^.*\\b1062\\b.*Duplicate entry '(?P<value>.*)' for key '(?P<columns>[^']+)'.*$")
@filters('mysql', sqla_exc.IntegrityError, "^.*\\b1062\\b.*Duplicate entry \\\\'(?P<value>.*)\\\\' for key \\\\'(?P<columns>.+)\\\\'.*$")
@filters('postgresql', sqla_exc.IntegrityError, ('^.*duplicate\\s+key.*"(?P<columns>[^"]+)"\\s*\\n.*Key\\s+\\((?P<key>.*)\\)=\\((?P<value>.*)\\)\\s+already\\s+exists.*$', '^.*duplicate\\s+key.*\\"(?P<columns>[^\\"]+)\\"\\s*\\n.*$'))
def _default_dupe_key_error(integrity_error, match, engine_name, is_disconnect):
    """Filter for MySQL or Postgresql duplicate key error.

    note(boris-42): In current versions of DB backends unique constraint
    violation messages follow the structure:

    postgres:
    1 column - (IntegrityError) duplicate key value violates unique
               constraint "users_c1_key"
    N columns - (IntegrityError) duplicate key value violates unique
               constraint "name_of_our_constraint"

    mysql since 8.0.19:
    1 column - (IntegrityError) (1062, "Duplicate entry 'value_of_c1' for key
               'table_name.c1'")
    N columns - (IntegrityError) (1062, "Duplicate entry 'values joined
               with -' for key 'table_name.name_of_our_constraint'")

    mysql+mysqldb:
    1 column - (IntegrityError) (1062, "Duplicate entry 'value_of_c1' for key
               'c1'")
    N columns - (IntegrityError) (1062, "Duplicate entry 'values joined
               with -' for key 'name_of_our_constraint'")

    mysql+mysqlconnector:
    1 column - (IntegrityError) 1062 (23000): Duplicate entry 'value_of_c1' for
               key 'c1'
    N columns - (IntegrityError) 1062 (23000): Duplicate entry 'values
               joined with -' for key 'name_of_our_constraint'
    """
    columns = match.group('columns')
    uniqbase = 'uniq_'
    if not columns.startswith(uniqbase):
        if engine_name == 'postgresql':
            columns = [columns[columns.index('_') + 1:columns.rindex('_')]]
        elif engine_name == 'mysql' and uniqbase in str(columns.split('0')[:1]):
            columns = columns.split('0')[1:]
        else:
            columns = [columns]
    else:
        columns = columns[len(uniqbase):].split('0')[1:]
    value = match.groupdict().get('value')
    raise exception.DBDuplicateEntry(columns, integrity_error, value)