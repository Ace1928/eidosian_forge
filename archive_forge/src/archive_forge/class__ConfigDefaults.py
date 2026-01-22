import os
import re
import warnings
from googlecloudsdk.third_party.appengine.api import lib_config
class _ConfigDefaults(object):

    def default_namespace_for_request():
        return None