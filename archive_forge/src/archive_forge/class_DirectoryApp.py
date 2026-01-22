import mimetypes
import os
from webob import exc
from webob.dec import wsgify
from webob.response import Response
class DirectoryApp(object):
    """An application that serves up the files in a given directory.

    This will serve index files (by default ``index.html``), or set
    ``index_page=None`` to disable this.  If you set
    ``hide_index_with_redirect=True`` (it defaults to False) then
    requests to, e.g., ``/index.html`` will be redirected to ``/``.

    To customize `FileApp` instances creation (which is what actually
    serves the responses), override the `make_fileapp` method.
    """

    def __init__(self, path, index_page='index.html', hide_index_with_redirect=False, **kw):
        self.path = os.path.abspath(path)
        if not self.path.endswith(os.path.sep):
            self.path += os.path.sep
        if not os.path.isdir(self.path):
            raise IOError('Path does not exist or is not directory: %r' % self.path)
        self.index_page = index_page
        self.hide_index_with_redirect = hide_index_with_redirect
        self.fileapp_kw = kw

    def make_fileapp(self, path):
        return FileApp(path, **self.fileapp_kw)

    @wsgify
    def __call__(self, req):
        path = os.path.abspath(os.path.join(self.path, req.path_info.lstrip('/')))
        if os.path.isdir(path) and self.index_page:
            return self.index(req, path)
        if self.index_page and self.hide_index_with_redirect and path.endswith(os.path.sep + self.index_page):
            new_url = req.path_url.rsplit('/', 1)[0]
            new_url += '/'
            if req.query_string:
                new_url += '?' + req.query_string
            return Response(status=301, location=new_url)
        if not path.startswith(self.path):
            return exc.HTTPForbidden()
        elif not os.path.isfile(path):
            return exc.HTTPNotFound(comment=path)
        else:
            return self.make_fileapp(path)

    def index(self, req, path):
        index_path = os.path.join(path, self.index_page)
        if not os.path.isfile(index_path):
            return exc.HTTPNotFound(comment=index_path)
        if not req.path_info.endswith('/'):
            url = req.path_url + '/'
            if req.query_string:
                url += '?' + req.query_string
            return Response(status=301, location=url)
        return self.make_fileapp(index_path)