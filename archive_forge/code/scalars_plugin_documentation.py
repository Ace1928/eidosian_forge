import csv
import io
import werkzeug.exceptions
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import metadata
Given a tag and list of runs, return dict of ScalarEvent arrays.