class EnvironForward(ForwardRequestExceptionMiddleware):

    def __call__(self, environ_, start_response):
        return self.app(environ, start_response)