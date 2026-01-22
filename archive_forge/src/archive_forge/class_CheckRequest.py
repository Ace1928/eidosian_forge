import random
import email.message
import pyzor
class CheckRequest(SimpleDigestBasedRequest):
    op = 'check'