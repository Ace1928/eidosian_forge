from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import subprocess
import sys
import threading
from . import comm
import ruamel.yaml as yaml
from six.moves import input
def MaybeWriteAppYaml(self):
    """Generates the app.yaml file if it doesn't already exist."""
    if not self.generated_appinfo:
        return
    notify = logging.info if self.params.deploy else self.env.Print
    filename = os.path.join(self.path, 'app.yaml')
    if self.params.appinfo or os.path.exists(filename):
        notify(FILE_EXISTS_MESSAGE.format('app.yaml'))
        return
    notify(WRITING_FILE_MESSAGE.format('app.yaml', self.path))
    with open(filename, 'w') as f:
        yaml.safe_dump(self.generated_appinfo, f, default_flow_style=False)