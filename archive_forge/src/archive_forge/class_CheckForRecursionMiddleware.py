class CheckForRecursionMiddleware(object):

    def __init__(self, app, env):
        self.app = app
        self.env = env

    def __call__(self, environ, start_response):
        path_info = environ.get('PATH_INFO', '')
        if path_info in self.env.get('pecan.recursive.old_path_info', []):
            raise RecursionLoop('Forwarding loop detected; %r visited twice (internal redirect path: %s)' % (path_info, self.env['pecan.recursive.old_path_info']))
        old_path_info = self.env.setdefault('pecan.recursive.old_path_info', [])
        old_path_info.append(self.env.get('PATH_INFO', ''))
        return self.app(environ, start_response)