from __future__ import absolute_import, division, print_function
import json
import socket
import getpass
from datetime import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.urls import open_url
from ansible.plugins.callback import CallbackBase

    ansible grafana callback plugin
    ansible.cfg:
        callback_plugins   = <path_to_callback_plugins_folder>
        callback_whitelist = grafana_annotations
    and put the plugin in <path_to_callback_plugins_folder>
    