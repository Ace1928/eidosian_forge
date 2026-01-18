from cheroot.wsgi import PathInfoDispatcher
def wsgi_invoke(app, environ):
    """Serve 1 request from a WSGI application."""
    response = {}

    def start_response(status, headers):
        response.update({'status': status, 'headers': headers})
    response['body'] = b''.join(app(environ, start_response))
    return response