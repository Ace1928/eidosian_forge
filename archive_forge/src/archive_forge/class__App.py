from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class _App(_ObjectType):
    name = 'application'
    egg_protocols = ['paste.app_factory', 'paste.composite_factory', 'paste.composit_factory']
    config_prefixes = [['app', 'application'], ['composite', 'composit'], 'pipeline', 'filter-app']

    def invoke(self, context):
        if context.protocol in ('paste.composit_factory', 'paste.composite_factory'):
            return fix_call(context.object, context.loader, context.global_conf, **context.local_conf)
        elif context.protocol == 'paste.app_factory':
            return fix_call(context.object, context.global_conf, **context.local_conf)
        else:
            assert 0, 'Protocol %r unknown' % context.protocol