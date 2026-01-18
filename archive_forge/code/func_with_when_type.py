from datetime import datetime, date
from decimal import Decimal
from json import JSONEncoder
def with_when_type(f):
    f.when_type = f.register
    return f