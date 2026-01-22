from gunicorn.http.message import Request
from gunicorn.http.unreader import SocketUnreader, IterUnreader
class RequestParser(Parser):
    mesg_class = Request