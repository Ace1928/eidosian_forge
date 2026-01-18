import os
import re
import sys
def unix2dos(file):
    """Replace LF with CRLF in argument files.  Print names of changed files."""
    if os.path.isdir(file):
        print(file, 'Directory!')
        return
    with open(file, 'rb') as fp:
        data = fp.read()
    if '\x00' in data:
        print(file, 'Binary!')
        return
    newdata = re.sub('\r\n', '\n', data)
    newdata = re.sub('\n', '\r\n', newdata)
    if newdata != data:
        print('unix2dos:', file)
        with open(file, 'wb') as f:
            f.write(newdata)
        return file
    else:
        print(file, 'ok')