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
def _create_sender_factory(self, to_addrs, msg, d):
    from twisted.mail.smtp import ESMTPSenderFactory
    factory_keywords = {'heloFallback': True, 'requireAuthentication': False, 'requireTransportSecurity': self.smtptls}
    if twisted_version >= Version('twisted', 21, 2, 0):
        factory_keywords['hostname'] = self.smtphost
    factory = ESMTPSenderFactory(self.smtpuser, self.smtppass, self.mailfrom, to_addrs, msg, d, **factory_keywords)
    factory.noisy = False
    return factory