from twisted.python import roots
from twisted.web import pages, resource
class VHostMonsterResource(resource.Resource):
    """
    Use this to be able to record the hostname and method (http vs. https)
    in the URL without disturbing your web site. If you put this resource
    in a URL http://foo.com/bar then requests to
    http://foo.com/bar/http/baz.com/something will be equivalent to
    http://foo.com/something, except that the hostname the request will
    appear to be accessing will be "baz.com". So if "baz.com" is redirecting
    all requests for to foo.com, while foo.com is inaccessible from the outside,
    then redirect and url generation will work correctly
    """

    def getChild(self, path, request):
        if path == b'http':
            request.isSecure = lambda: 0
        elif path == b'https':
            request.isSecure = lambda: 1
        return _HostResource()