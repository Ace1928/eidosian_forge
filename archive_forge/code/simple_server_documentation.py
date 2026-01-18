from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import urllib.parse
from wsgiref.handlers import SimpleHandler
from platform import python_implementation
Handle a single HTTP request