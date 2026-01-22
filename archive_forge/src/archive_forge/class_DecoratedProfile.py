import sys
import os
import hotshot
import hotshot.stats
import threading
import cgi
import time
from io import StringIO
from paste import response
class DecoratedProfile(object):
    lock = threading.Lock()

    def __init__(self, func, **options):
        self.func = func
        self.options = options

    def __call__(self, *args, **kw):
        self.lock.acquire()
        try:
            return self.profile(self.func, *args, **kw)
        finally:
            self.lock.release()

    def profile(self, func, *args, **kw):
        ops = self.options
        prof_filename = ops.get('log_filename', 'profile_data.log.tmp')
        prof = hotshot.Profile(prof_filename)
        prof.addinfo('Function Call', self.format_function(func, *args, **kw))
        if ops.get('add_info'):
            prof.addinfo('Extra info', ops['add_info'])
        exc_info = None
        try:
            start_time = time.time()
            try:
                result = prof.runcall(func, *args, **kw)
            except:
                exc_info = sys.exc_info()
            end_time = time.time()
        finally:
            prof.close()
        stats = hotshot.stats.load(prof_filename)
        os.unlink(prof_filename)
        if ops.get('strip_dirs', True):
            stats.strip_dirs()
        stats.sort_stats(*ops.get('sort_stats', ('time', 'calls')))
        display_limit = ops.get('display_limit', 20)
        output = capture_output(stats.print_stats, display_limit)
        output_callers = capture_output(stats.print_callers, display_limit)
        output_file = ops.get('log_file')
        if output_file in (None, 'stderr'):
            f = sys.stderr
        elif output_file in ('-', 'stdout'):
            f = sys.stdout
        else:
            f = open(output_file, 'a')
            f.write('\n%s\n' % ('-' * 60))
            f.write('Date: %s\n' % time.strftime('%c'))
        f.write('Function call: %s\n' % self.format_function(func, *args, **kw))
        f.write('Wall time: %0.2f seconds\n' % (end_time - start_time))
        f.write(output)
        f.write(output_callers)
        if output_file not in (None, '-', 'stdout', 'stderr'):
            f.close()
        if exc_info:
            raise exc_info
        return result

    def format_function(self, func, *args, **kw):
        args = map(repr, args)
        args.extend(['%s=%r' % (k, v) for k, v in kw.items()])
        return '%s(%s)' % (func.__name__, ', '.join(args))