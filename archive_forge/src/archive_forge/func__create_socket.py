import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _create_socket(self, timeout):
    sock = IMAP4._create_socket(self, timeout)
    return self.ssl_context.wrap_socket(sock, server_hostname=self.host)