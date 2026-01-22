from io import BytesIO
from ...bzr.smart import medium
from ...transport import chroot, get_transport
from ...urlutils import local_path_to_url
class RelpathSetter:
    """WSGI middleware to set 'breezy.relpath' in the environ.

    Different servers can invoke a SmartWSGIApp in different ways.  This
    middleware allows an adminstrator to configure how to the SmartWSGIApp will
    determine what path it should be serving for a given request for many common
    situations.

    For example, a request for "/some/prefix/repo/branch/.bzr/smart" received by
    a typical Apache and mod_fastcgi configuration will set `REQUEST_URI` to
    "/some/prefix/repo/branch/.bzr/smart".  A RelpathSetter with
    prefix="/some/prefix/" and path_var="REQUEST_URI" will set that request's
    'breezy.relpath' variable to "repo/branch".
    """

    def __init__(self, app, prefix='', path_var='REQUEST_URI'):
        """Constructor.

        :param app: WSGI app to wrap, e.g. a SmartWSGIApp instance.
        :param path_var: the variable in the WSGI environ to calculate the
            'breezy.relpath' variable from.
        :param prefix: a prefix to strip from the variable specified in
            path_var before setting 'breezy.relpath'.
        """
        self.app = app
        self.prefix = prefix
        self.path_var = path_var

    def __call__(self, environ, start_response):
        path = environ[self.path_var]
        suffix = '/.bzr/smart'
        if not (path.startswith(self.prefix) and path.endswith(suffix)):
            start_response('404 Not Found', [])
            return []
        environ['breezy.relpath'] = path[len(self.prefix):-len(suffix)]
        return self.app(environ, start_response)