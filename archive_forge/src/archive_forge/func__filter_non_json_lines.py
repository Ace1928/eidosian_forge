from __future__ import (absolute_import, division, print_function)
import json  # pylint: disable=unused-import
def _filter_non_json_lines(data, objects_only=False):
    """
    Used to filter unrelated output around module JSON output, like messages from
    tcagetattr, or where dropbear spews MOTD on every single command (which is nuts).

    Filters leading lines before first line-starting occurrence of '{' or '[', and filter all
    trailing lines after matching close character (working from the bottom of output).
    """
    warnings = []
    lines = data.splitlines()
    for start, line in enumerate(lines):
        line = line.strip()
        if line.startswith(u'{'):
            endchar = u'}'
            break
        elif not objects_only and line.startswith(u'['):
            endchar = u']'
            break
    else:
        raise ValueError('No start of json char found')
    lines = lines[start:]
    for reverse_end_offset, line in enumerate(reversed(lines)):
        if line.strip().endswith(endchar):
            break
    else:
        raise ValueError('No end of json char found')
    if reverse_end_offset > 0:
        trailing_junk = lines[len(lines) - reverse_end_offset:]
        for line in trailing_junk:
            if line.strip():
                warnings.append('Module invocation had junk after the JSON data: %s' % '\n'.join(trailing_junk))
                break
    lines = lines[:len(lines) - reverse_end_offset]
    return ('\n'.join(lines), warnings)