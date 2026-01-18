import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_smtp_password_from_user(self):
    user = 'joe'
    password = 'hispass'
    factory = WideOpenSMTPFactory()
    conn = self.get_connection(b'[DEFAULT]\nsmtp_username=%s\n' % user.encode('ascii'), smtp_factory=factory)
    self.assertIs(None, conn._smtp_password)
    ui.ui_factory = ui.CannedInputUIFactory([password])
    conn._connect()
    self.assertEqual(password, conn._smtp_password)