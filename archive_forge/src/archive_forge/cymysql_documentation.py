from .base import BIT
from .base import MySQLDialect
from .mysqldb import MySQLDialect_mysqldb
from ... import util
Convert MySQL's 64 bit, variable length binary string to a long.