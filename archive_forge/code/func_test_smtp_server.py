import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_smtp_server(self):
    conn = self.get_connection(b'smtp_server=host:10')
    self.assertEqual('host:10', conn._smtp_server)