from time import localtime
from datetime import date, datetime, time, timedelta
from MySQLdb._mysql import string_literal
def Date_or_None(s):
    try:
        return date(int(s[:4]), int(s[5:7]), int(s[8:10]))
    except ValueError:
        return None