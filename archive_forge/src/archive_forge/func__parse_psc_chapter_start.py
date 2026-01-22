import datetime
import re
from .. import util
def _parse_psc_chapter_start(start):
    m = format_.match(start)
    if m is None:
        return None
    _, h, m, s, _, ms = m.groups()
    h, m, s, ms = (int(h or 0), int(m), int(s), int(ms or 0))
    return datetime.timedelta(0, h * 60 * 60 + m * 60 + s, ms * 1000)