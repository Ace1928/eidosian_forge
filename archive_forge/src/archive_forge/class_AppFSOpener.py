from __future__ import absolute_import, print_function, unicode_literals
import typing
from .base import Opener
from .errors import OpenerError
from .registry import registry
@registry.install
class AppFSOpener(Opener):
    """``AppFS`` opener."""
    protocols = ['userdata', 'userconf', 'sitedata', 'siteconf', 'usercache', 'userlog']
    _protocol_mapping = None

    def open_fs(self, fs_url, parse_result, writeable, create, cwd):
        from .. import appfs
        from ..subfs import ClosingSubFS
        if self._protocol_mapping is None:
            self._protocol_mapping = {'userdata': appfs.UserDataFS, 'userconf': appfs.UserConfigFS, 'sitedata': appfs.SiteDataFS, 'siteconf': appfs.SiteConfigFS, 'usercache': appfs.UserCacheFS, 'userlog': appfs.UserLogFS}
        fs_class = self._protocol_mapping[parse_result.protocol]
        resource, delim, path = parse_result.resource.partition('/')
        tokens = resource.split(':', 3)
        if len(tokens) == 2:
            appname, author = tokens
            version = None
        elif len(tokens) == 3:
            appname, author, version = tokens
        else:
            raise OpenerError('resource should be <appname>:<author> or <appname>:<author>:<version>')
        app_fs = fs_class(appname, author=author, version=version, create=create)
        if delim:
            if create:
                app_fs.makedir(path, recreate=True)
            return app_fs.opendir(path, factory=ClosingSubFS)
        return app_fs