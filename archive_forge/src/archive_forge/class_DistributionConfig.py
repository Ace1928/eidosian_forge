import uuid
import base64
import time
from boto.compat import six, json
from boto.cloudfront.identity import OriginAccessIdentity
from boto.cloudfront.object import Object, StreamingObject
from boto.cloudfront.signers import ActiveTrustedSigners, TrustedSigners
from boto.cloudfront.logging import LoggingInfo
from boto.cloudfront.origin import S3Origin, CustomOrigin
from boto.s3.acl import ACL
class DistributionConfig(object):

    def __init__(self, connection=None, origin=None, enabled=False, caller_reference='', cnames=None, comment='', trusted_signers=None, default_root_object=None, logging=None):
        """
        :param origin: Origin information to associate with the
                       distribution.  If your distribution will use
                       an Amazon S3 origin, then this should be an
                       S3Origin object. If your distribution will use
                       a custom origin (non Amazon S3), then this
                       should be a CustomOrigin object.
        :type origin: :class:`boto.cloudfront.origin.S3Origin` or
                      :class:`boto.cloudfront.origin.CustomOrigin`

        :param enabled: Whether the distribution is enabled to accept
                        end user requests for content.
        :type enabled: bool

        :param caller_reference: A unique number that ensures the
                                 request can't be replayed.  If no
                                 caller_reference is provided, boto
                                 will generate a type 4 UUID for use
                                 as the caller reference.
        :type enabled: str

        :param cnames: A CNAME alias you want to associate with this
                       distribution. You can have up to 10 CNAME aliases
                       per distribution.
        :type enabled: array of str

        :param comment: Any comments you want to include about the
                        distribution.
        :type comment: str

        :param trusted_signers: Specifies any AWS accounts you want to
                                permit to create signed URLs for private
                                content. If you want the distribution to
                                use signed URLs, this should contain a
                                TrustedSigners object; if you want the
                                distribution to use basic URLs, leave
                                this None.
        :type trusted_signers: :class`boto.cloudfront.signers.TrustedSigners`

        :param default_root_object: Designates a default root object.
                                    Only include a DefaultRootObject value
                                    if you are going to assign a default
                                    root object for the distribution.
        :type comment: str

        :param logging: Controls whether access logs are written for the
                        distribution. If you want to turn on access logs,
                        this should contain a LoggingInfo object; otherwise
                        it should contain None.
        :type logging: :class`boto.cloudfront.logging.LoggingInfo`

        """
        self.connection = connection
        self.origin = origin
        self.enabled = enabled
        if caller_reference:
            self.caller_reference = caller_reference
        else:
            self.caller_reference = str(uuid.uuid4())
        self.cnames = []
        if cnames:
            self.cnames = cnames
        self.comment = comment
        self.trusted_signers = trusted_signers
        self.logging = logging
        self.default_root_object = default_root_object

    def __repr__(self):
        return 'DistributionConfig:%s' % self.origin

    def to_xml(self):
        s = '<?xml version="1.0" encoding="UTF-8"?>\n'
        s += '<DistributionConfig xmlns="http://cloudfront.amazonaws.com/doc/2010-07-15/">\n'
        if self.origin:
            s += self.origin.to_xml()
        s += '  <CallerReference>%s</CallerReference>\n' % self.caller_reference
        for cname in self.cnames:
            s += '  <CNAME>%s</CNAME>\n' % cname
        if self.comment:
            s += '  <Comment>%s</Comment>\n' % self.comment
        s += '  <Enabled>'
        if self.enabled:
            s += 'true'
        else:
            s += 'false'
        s += '</Enabled>\n'
        if self.trusted_signers:
            s += '<TrustedSigners>\n'
            for signer in self.trusted_signers:
                if signer == 'Self':
                    s += '  <Self></Self>\n'
                else:
                    s += '  <AwsAccountNumber>%s</AwsAccountNumber>\n' % signer
            s += '</TrustedSigners>\n'
        if self.logging:
            s += '<Logging>\n'
            s += '  <Bucket>%s</Bucket>\n' % self.logging.bucket
            s += '  <Prefix>%s</Prefix>\n' % self.logging.prefix
            s += '</Logging>\n'
        if self.default_root_object:
            dro = self.default_root_object
            s += '<DefaultRootObject>%s</DefaultRootObject>\n' % dro
        s += '</DistributionConfig>\n'
        return s

    def startElement(self, name, attrs, connection):
        if name == 'TrustedSigners':
            self.trusted_signers = TrustedSigners()
            return self.trusted_signers
        elif name == 'Logging':
            self.logging = LoggingInfo()
            return self.logging
        elif name == 'S3Origin':
            self.origin = S3Origin()
            return self.origin
        elif name == 'CustomOrigin':
            self.origin = CustomOrigin()
            return self.origin
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'CNAME':
            self.cnames.append(value)
        elif name == 'Comment':
            self.comment = value
        elif name == 'Enabled':
            if value.lower() == 'true':
                self.enabled = True
            else:
                self.enabled = False
        elif name == 'CallerReference':
            self.caller_reference = value
        elif name == 'DefaultRootObject':
            self.default_root_object = value
        else:
            setattr(self, name, value)