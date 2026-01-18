import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
A fake smtp server that implements login by accepting anybody.