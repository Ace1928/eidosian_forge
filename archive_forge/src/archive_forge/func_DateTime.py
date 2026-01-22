from oslo_log import log as logging
import sqlalchemy.types
def DateTime():
    return sqlalchemy.types.DateTime(timezone=False)