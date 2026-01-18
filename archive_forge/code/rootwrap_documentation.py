import os
import signal
import threading
import time
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import strutils
from os_brick import exception
from os_brick import privileged
Create a symbolic link with sys admin privileges.

    This method behaves like the "ln -s" command, including the force parameter
    where it will replace the link_name file even if it's not a symlink.
    