import logging
from email import encoders as Encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from io import BytesIO
from twisted import version as twisted_version
from twisted.internet import defer, ssl
from twisted.python.versions import Version
from scrapy.utils.misc import arg_to_iter
from scrapy.utils.python import to_bytes
def _sent_failed(self, failure, to, cc, subject, nattachs):
    errstr = str(failure.value)
    logger.error('Unable to send mail: To=%(mailto)s Cc=%(mailcc)s Subject="%(mailsubject)s" Attachs=%(mailattachs)d- %(mailerr)s', {'mailto': to, 'mailcc': cc, 'mailsubject': subject, 'mailattachs': nattachs, 'mailerr': errstr})
    return failure