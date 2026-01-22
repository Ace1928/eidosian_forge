import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
class BasicAppWrapper:

    def __init__(self, app):
        self.app = app
        self.app.withdraw()