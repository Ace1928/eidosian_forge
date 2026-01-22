import re
class ExperimentIdMiddleware:
    """WSGI middleware extracting experiment IDs from URL to environment.

    Any request whose path matches `/experiment/SOME_EID[/...]` will have
    its first two path components stripped, and its experiment ID stored
    onto the WSGI environment with key taken from the `WSGI_ENVIRON_KEY`
    constant. All other requests will have paths unchanged and the
    experiment ID set to the empty string. It noops if the key taken from
    the `WSGI_ENVIRON_KEY` is already present in the environment.

    Instances of this class are WSGI applications (see PEP 3333).
    """

    def __init__(self, application):
        """Initializes an `ExperimentIdMiddleware`.

        Args:
          application: The WSGI application to wrap (see PEP 3333).
        """
        self._application = application
        self._pat = re.compile('/%s/([^/]*)' % re.escape(_EXPERIMENT_PATH_COMPONENT))

    def __call__(self, environ, start_response):
        if WSGI_ENVIRON_KEY in environ:
            return self._application(environ, start_response)
        path = environ.get('PATH_INFO', '')
        m = self._pat.match(path)
        if m:
            eid = m.group(1)
            new_path = path[m.end(0):]
            root = m.group(0)
        else:
            eid = ''
            new_path = path
            root = ''
        environ[WSGI_ENVIRON_KEY] = eid
        environ['PATH_INFO'] = new_path
        environ['SCRIPT_NAME'] = environ.get('SCRIPT_NAME', '') + root
        return self._application(environ, start_response)