import threading
import socket
import select
@classmethod
def text_response_server(cls, text, request_timeout=0.5, **kwargs):

    def text_response_handler(sock):
        request_content = consume_socket_content(sock, timeout=request_timeout)
        sock.send(text.encode('utf-8'))
        return request_content
    return Server(text_response_handler, **kwargs)