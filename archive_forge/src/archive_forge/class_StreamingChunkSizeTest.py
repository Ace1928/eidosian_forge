from tornado import gen, netutil
from tornado.escape import (
from tornado.http1connection import HTTP1Connection
from tornado.httpclient import HTTPError
from tornado.httpserver import HTTPServer
from tornado.httputil import (
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import ssl_options_to_context
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import (
from tornado.test.util import skipOnTravis
from tornado.web import Application, RequestHandler, stream_request_body
from contextlib import closing
import datetime
import gzip
import logging
import os
import shutil
import socket
import ssl
import sys
import tempfile
import textwrap
import unittest
import urllib.parse
from io import BytesIO
import typing
class StreamingChunkSizeTest(AsyncHTTPTestCase):
    BODY = b'01234567890123456789012345678901234567890123456789'
    CHUNK_SIZE = 16

    def get_http_client(self):
        return SimpleAsyncHTTPClient()

    def get_httpserver_options(self):
        return dict(chunk_size=self.CHUNK_SIZE, decompress_request=True)

    class MessageDelegate(HTTPMessageDelegate):

        def __init__(self, connection):
            self.connection = connection

        def headers_received(self, start_line, headers):
            self.chunk_lengths = []

        def data_received(self, chunk):
            self.chunk_lengths.append(len(chunk))

        def finish(self):
            response_body = utf8(json_encode(self.chunk_lengths))
            self.connection.write_headers(ResponseStartLine('HTTP/1.1', 200, 'OK'), HTTPHeaders({'Content-Length': str(len(response_body))}))
            self.connection.write(response_body)
            self.connection.finish()

    def get_app(self):

        class App(HTTPServerConnectionDelegate):

            def start_request(self, server_conn, request_conn):
                return StreamingChunkSizeTest.MessageDelegate(request_conn)
        return App()

    def fetch_chunk_sizes(self, **kwargs):
        response = self.fetch('/', method='POST', **kwargs)
        response.rethrow()
        chunks = json_decode(response.body)
        self.assertEqual(len(self.BODY), sum(chunks))
        for chunk_size in chunks:
            self.assertLessEqual(chunk_size, self.CHUNK_SIZE, 'oversized chunk: ' + str(chunks))
            self.assertGreater(chunk_size, 0, 'empty chunk: ' + str(chunks))
        return chunks

    def compress(self, body):
        bytesio = BytesIO()
        gzfile = gzip.GzipFile(mode='w', fileobj=bytesio)
        gzfile.write(body)
        gzfile.close()
        compressed = bytesio.getvalue()
        if len(compressed) >= len(body):
            raise Exception('body did not shrink when compressed')
        return compressed

    def test_regular_body(self):
        chunks = self.fetch_chunk_sizes(body=self.BODY)
        self.assertEqual([16, 16, 16, 2], chunks)

    def test_compressed_body(self):
        self.fetch_chunk_sizes(body=self.compress(self.BODY), headers={'Content-Encoding': 'gzip'})

    def test_chunked_body(self):

        def body_producer(write):
            write(self.BODY[:20])
            write(self.BODY[20:])
        chunks = self.fetch_chunk_sizes(body_producer=body_producer)
        self.assertEqual([16, 4, 16, 14], chunks)

    def test_chunked_compressed(self):
        compressed = self.compress(self.BODY)
        self.assertGreater(len(compressed), 20)

        def body_producer(write):
            write(compressed[:20])
            write(compressed[20:])
        self.fetch_chunk_sizes(body_producer=body_producer, headers={'Content-Encoding': 'gzip'})