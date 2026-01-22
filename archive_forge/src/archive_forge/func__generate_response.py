import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def _generate_response(self, path):
    local_path = self.translate_path(path)
    st = os.stat(local_path)
    prop = dict()

    def _prop(ns, name, value=None):
        if value is None:
            return '<{}:{}/>'.format(ns, name)
        else:
            return '<{}:{}>{}</{}:{}>'.format(ns, name, value, ns, name)
    if stat.S_ISDIR(st.st_mode):
        dpath = path
        if not dpath.endswith('/'):
            dpath += '/'
        prop['href'] = _prop('D', 'href', dpath)
        prop['type'] = _prop('liveprop', 'resourcetype', '<D:collection/>')
        prop['length'] = ''
        prop['exec'] = ''
    else:
        prop['href'] = _prop('D', 'href', path)
        prop['type'] = _prop('liveprop', 'resourcetype')
        prop['length'] = _prop('liveprop', 'getcontentlength', st.st_size)
        if st.st_mode & stat.S_IXUSR:
            is_exec = 'T'
        else:
            is_exec = 'F'
        prop['exec'] = _prop('bzr', 'executable', is_exec)
    prop['status'] = _prop('D', 'status', 'HTTP/1.1 200 OK')
    response = '<D:response xmlns:liveprop="DAV:" xmlns:bzr="DAV:">\n    %(href)s\n    <D:propstat>\n        <D:prop>\n             %(type)s\n             %(length)s\n             %(exec)s\n        </D:prop>\n        %(status)s\n    </D:propstat>\n</D:response>\n' % prop
    return (response, st)