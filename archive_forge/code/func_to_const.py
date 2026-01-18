import re
def to_const(string):
    return re.sub('[\\W|^]+', '_', string).upper()