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

                        Currently only single disk image extraction is
                        supported.
                        FIXME(dramakri): Support multiple images in OVA package
                        