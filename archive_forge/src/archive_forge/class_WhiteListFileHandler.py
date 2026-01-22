import os
import re
import tornado.web
from .paths import collect_static_paths
class WhiteListFileHandler(tornado.web.StaticFileHandler):

    def initialize(self, whitelist=[], blacklist=[], **kwargs):
        self.whitelist = whitelist
        self.blacklist = blacklist
        super(WhiteListFileHandler, self).initialize(**kwargs)

    def get_absolute_path(self, root, path):
        whitelisted = any((re.fullmatch(pattern, path) for pattern in self.whitelist))
        blacklisted = any((re.fullmatch(pattern, path) for pattern in self.blacklist))
        if not whitelisted:
            raise tornado.web.HTTPError(403, 'File not whitelisted')
        if blacklisted:
            raise tornado.web.HTTPError(403, 'File blacklisted')
        return super(WhiteListFileHandler, self).get_absolute_path(root, path)