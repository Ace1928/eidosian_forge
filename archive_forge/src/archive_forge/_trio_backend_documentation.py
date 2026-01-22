import socket
import trio
import trio.socket  # type: ignore
import dns._asyncbackend
import dns._features
import dns.exception
import dns.inet
trio async I/O library query support