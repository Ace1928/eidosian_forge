from time import localtime
from datetime import date, datetime, time, timedelta
from MySQLdb._mysql import string_literal
def DateTime2literal(d, c):
    """Format a DateTime object as an ISO timestamp."""
    return string_literal(format_TIMESTAMP(d))