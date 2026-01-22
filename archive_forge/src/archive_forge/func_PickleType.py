from oslo_log import log as logging
import sqlalchemy.types
def PickleType():
    return sqlalchemy.types.PickleType()