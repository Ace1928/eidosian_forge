from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class _PipeLine(_ObjectType):
    name = 'pipeline'

    def invoke(self, context):
        app = context.app_context.create()
        filters = [c.create() for c in context.filter_contexts]
        filters.reverse()
        for filter in filters:
            app = filter(app)
        return app