import io
import os
import os.path
import sys
import warnings
import cherrypy
class ProfileAggregator(Profiler):

    def __init__(self, path=None):
        Profiler.__init__(self, path)
        global _count
        self.count = _count = _count + 1
        self.profiler = profile.Profile()

    def run(self, func, *args, **params):
        path = os.path.join(self.path, 'cp_%04d.prof' % self.count)
        result = self.profiler.runcall(func, *args, **params)
        self.profiler.dump_stats(path)
        return result