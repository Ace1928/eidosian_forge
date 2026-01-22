import gzip
import os
import shutil
import zipfile
from oslo_log import log as logging
from oslo_utils import encodeutils
from taskflow.patterns import linear_flow as lf
from taskflow import task
def _lhafile(src_path, dest_path, image_id):
    if NO_LHA:
        raise Exception('No lhafile available.')
    try:
        with lhafile.LhaFile(src_path, 'r') as lfd:
            content = lfd.namelist()
            if len(content) != 1:
                raise Exception('Archive contains more than one file.')
            else:
                lfd.extract(content[0], dest_path)
    except Exception as e:
        LOG.debug('LHA: Error decompressing image %(iid)s: %(msg)s', {'iid': image_id, 'msg': encodeutils.exception_to_unicode(e)})
        raise