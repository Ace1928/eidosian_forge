import cherrypy
from cherrypy import _json as json
from cherrypy._cpcompat import text_or_bytes, ntou
Wrap request.handler to serialize its output to JSON. Sets Content-Type.

    If the given content_type is None, the Content-Type response header
    is not set.

    Provide your own handler to use a custom encoder.  For example
    cherrypy.config['tools.json_out.handler'] = <function>, or
    @json_out(handler=function).
    