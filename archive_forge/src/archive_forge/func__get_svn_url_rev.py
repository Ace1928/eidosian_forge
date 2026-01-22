import logging
import os
import re
from typing import List, Optional, Tuple
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import CommandArgs, make_command
from pip._internal.vcs.versioncontrol import (
@classmethod
def _get_svn_url_rev(cls, location: str) -> Tuple[Optional[str], int]:
    from pip._internal.exceptions import InstallationError
    entries_path = os.path.join(location, cls.dirname, 'entries')
    if os.path.exists(entries_path):
        with open(entries_path) as f:
            data = f.read()
    else:
        data = ''
    url = None
    if data.startswith('8') or data.startswith('9') or data.startswith('10'):
        entries = list(map(str.splitlines, data.split('\n\x0c\n')))
        del entries[0][0]
        url = entries[0][3]
        revs = [int(d[9]) for d in entries if len(d) > 9 and d[9]] + [0]
    elif data.startswith('<?xml'):
        match = _svn_xml_url_re.search(data)
        if not match:
            raise ValueError(f'Badly formatted data: {data!r}')
        url = match.group(1)
        revs = [int(m.group(1)) for m in _svn_rev_re.finditer(data)] + [0]
    else:
        try:
            xml = cls.run_command(['info', '--xml', location], show_stdout=False, stdout_only=True)
            match = _svn_info_xml_url_re.search(xml)
            assert match is not None
            url = match.group(1)
            revs = [int(m.group(1)) for m in _svn_info_xml_rev_re.finditer(xml)]
        except InstallationError:
            url, revs = (None, [])
    if revs:
        rev = max(revs)
    else:
        rev = 0
    return (url, rev)