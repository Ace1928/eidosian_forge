from boto.cloudfront.identity import OriginAccessIdentity
class CustomOrigin(object):
    """
    Origin information to associate with the distribution.
    If your distribution will use a non-Amazon S3 origin,
    then you use the CustomOrigin element.
    """

    def __init__(self, dns_name=None, http_port=80, https_port=443, origin_protocol_policy=None):
        """
        :param dns_name: The DNS name of your Amazon S3 bucket to
                         associate with the distribution.
                         For example: mybucket.s3.amazonaws.com.
        :type dns_name: str
        
        :param http_port: The HTTP port the custom origin listens on.
        :type http_port: int
        
        :param https_port: The HTTPS port the custom origin listens on.
        :type http_port: int
        
        :param origin_protocol_policy: The origin protocol policy to
                                       apply to your origin. If you
                                       specify http-only, CloudFront
                                       will use HTTP only to access the origin.
                                       If you specify match-viewer, CloudFront
                                       will fetch from your origin using HTTP
                                       or HTTPS, based on the protocol of the
                                       viewer request.
        :type origin_protocol_policy: str
        
        """
        self.dns_name = dns_name
        self.http_port = http_port
        self.https_port = https_port
        self.origin_protocol_policy = origin_protocol_policy

    def __repr__(self):
        return '<CustomOrigin: %s>' % self.dns_name

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'DNSName':
            self.dns_name = value
        elif name == 'HTTPPort':
            try:
                self.http_port = int(value)
            except ValueError:
                self.http_port = value
        elif name == 'HTTPSPort':
            try:
                self.https_port = int(value)
            except ValueError:
                self.https_port = value
        elif name == 'OriginProtocolPolicy':
            self.origin_protocol_policy = value
        else:
            setattr(self, name, value)

    def to_xml(self):
        s = '  <CustomOrigin>\n'
        s += '    <DNSName>%s</DNSName>\n' % self.dns_name
        s += '    <HTTPPort>%d</HTTPPort>\n' % self.http_port
        s += '    <HTTPSPort>%d</HTTPSPort>\n' % self.https_port
        s += '    <OriginProtocolPolicy>%s</OriginProtocolPolicy>\n' % self.origin_protocol_policy
        s += '  </CustomOrigin>\n'
        return s