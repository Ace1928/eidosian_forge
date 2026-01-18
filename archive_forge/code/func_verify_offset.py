from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def verify_offset(instring, parsed):
    time_epsilon = timedelta(seconds=1)
    if instring in expected:
        if parsed.time_offset - expected[instring] <= time_epsilon:
            parsed['verify_offset'] = 'PASS'
        else:
            parsed['verify_offset'] = 'FAIL'