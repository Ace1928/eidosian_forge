import dataclasses
import json
from jupyter_server.base.handlers import APIHandler
from tornado import web
from jupyterlab.extensions.manager import PluginManager
POST query performs an action on a specific plugin

        Body arguments:
            {
                "cmd": Action to perform - ["enable", "disable"]
                "plugin_name": Plugin name
            }
        