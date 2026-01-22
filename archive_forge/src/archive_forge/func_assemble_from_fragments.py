from __future__ import absolute_import, division, print_function
import codecs
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import b, indexbytes
from ansible.module_utils.common.text.converters import to_native
def assemble_from_fragments(src_path, delimiter=None, compiled_regexp=None, ignore_hidden=False, tmpdir=None):
    """ assemble a file from a directory of fragments """
    tmpfd, temp_path = tempfile.mkstemp(dir=tmpdir)
    tmp = os.fdopen(tmpfd, 'wb')
    delimit_me = False
    add_newline = False
    for f in sorted(os.listdir(src_path)):
        if compiled_regexp and (not compiled_regexp.search(f)):
            continue
        fragment = os.path.join(src_path, f)
        if not os.path.isfile(fragment) or (ignore_hidden and os.path.basename(fragment).startswith('.')):
            continue
        with open(fragment, 'rb') as fragment_fh:
            fragment_content = fragment_fh.read()
        if add_newline:
            tmp.write(b('\n'))
        if delimit_me:
            if delimiter:
                delimiter = codecs.escape_decode(delimiter)[0]
                tmp.write(delimiter)
                if indexbytes(delimiter, -1) != 10:
                    tmp.write(b('\n'))
        tmp.write(fragment_content)
        delimit_me = True
        if fragment_content.endswith(b('\n')):
            add_newline = False
        else:
            add_newline = True
    tmp.close()
    return temp_path