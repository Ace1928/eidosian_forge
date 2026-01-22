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
def _get_namespace_and_tag(tag):
    """Separate and return the namespace and tag elements.

            There is no native support for this operation in elementtree
            package. See http://bugs.python.org/issue18304 for details.
            """
    m = re.match('\\{(.+)\\}(.+)', tag)
    if m:
        return (m.group(1), m.group(2))
    else:
        return ('', tag)