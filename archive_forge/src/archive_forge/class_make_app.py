import io
import os
import os.path
import sys
import warnings
import cherrypy
class make_app:

    def __init__(self, nextapp, path=None, aggregate=False):
        """Make a WSGI middleware app which wraps 'nextapp' with profiling.

        nextapp
            the WSGI application to wrap, usually an instance of
            cherrypy.Application.

        path
            where to dump the profiling output.

        aggregate
            if True, profile data for all HTTP requests will go in
            a single file. If False (the default), each HTTP request will
            dump its profile data into a separate file.

        """
        if profile is None or pstats is None:
            msg = "Your installation of Python does not have a profile module. If you're on Debian, try `sudo apt-get install python-profiler`. See http://www.cherrypy.org/wiki/ProfilingOnDebian for details."
            warnings.warn(msg)
        self.nextapp = nextapp
        self.aggregate = aggregate
        if aggregate:
            self.profiler = ProfileAggregator(path)
        else:
            self.profiler = Profiler(path)

    def __call__(self, environ, start_response):

        def gather():
            result = []
            for line in self.nextapp(environ, start_response):
                result.append(line)
            return result
        return self.profiler.run(gather)