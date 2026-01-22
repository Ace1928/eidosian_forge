import logging
import re
from .compat import string_types
from .util import parse_requirement
def _pep_440_key(s):
    s = s.strip()
    m = PEP440_VERSION_RE.match(s)
    if not m:
        raise UnsupportedVersionError('Not a valid version: %s' % s)
    groups = m.groups()
    nums = tuple((int(v) for v in groups[1].split('.')))
    while len(nums) > 1 and nums[-1] == 0:
        nums = nums[:-1]
    if not groups[0]:
        epoch = 0
    else:
        epoch = int(groups[0][:-1])
    pre = groups[4:6]
    post = groups[7:9]
    dev = groups[10:12]
    local = groups[13]
    if pre == (None, None):
        pre = ()
    elif pre[1] is None:
        pre = (pre[0], 0)
    else:
        pre = (pre[0], int(pre[1]))
    if post == (None, None):
        post = ()
    elif post[1] is None:
        post = (post[0], 0)
    else:
        post = (post[0], int(post[1]))
    if dev == (None, None):
        dev = ()
    elif dev[1] is None:
        dev = (dev[0], 0)
    else:
        dev = (dev[0], int(dev[1]))
    if local is None:
        local = ()
    else:
        parts = []
        for part in local.split('.'):
            if part.isdigit():
                part = (1, int(part))
            else:
                part = (0, part)
            parts.append(part)
        local = tuple(parts)
    if not pre:
        if not post and dev:
            pre = ('a', -1)
        else:
            pre = ('z',)
    if not post:
        post = ('_',)
    if not dev:
        dev = ('final',)
    return (epoch, nums, pre, post, dev, local)