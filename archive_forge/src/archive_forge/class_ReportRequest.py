import random
import email.message
import pyzor
class ReportRequest(SimpleDigestSpecBasedRequest):
    op = 'report'