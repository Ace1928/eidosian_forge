import os
import re
import shutil
import tarfile
import urllib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from taskflow.patterns import linear_flow as lf
from taskflow import task
from glance.i18n import _, _LW
def _get_extracted_file_path(self, image_id):
    file_path = CONF.task.work_dir
    if CONF.enabled_backends:
        file_path = getattr(CONF, 'os_glance_tasks_store').filesystem_store_datadir
    return os.path.join(file_path, '%s.extracted' % image_id)