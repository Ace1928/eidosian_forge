from boto.cloudfront.identity import OriginAccessIdentity
class S3Origin(object):
    """
    Origin information to associate with the distribution.
    If your distribution will use an Amazon S3 origin,
    then you use the S3Origin element.
    """

    def __init__(self, dns_name=None, origin_access_identity=None):
        """
        :param dns_name: The DNS name of your Amazon S3 bucket to
                         associate with the distribution.
                         For example: mybucket.s3.amazonaws.com.
        :type dns_name: str
        
        :param origin_access_identity: The CloudFront origin access
                                       identity to associate with the
                                       distribution. If you want the
                                       distribution to serve private content,
                                       include this element; if you want the
                                       distribution to serve public content,
                                       remove this element.
        :type origin_access_identity: str
        
        """
        self.dns_name = dns_name
        self.origin_access_identity = origin_access_identity

    def __repr__(self):
        return '<S3Origin: %s>' % self.dns_name

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'DNSName':
            self.dns_name = value
        elif name == 'OriginAccessIdentity':
            self.origin_access_identity = value
        else:
            setattr(self, name, value)

    def to_xml(self):
        s = '  <S3Origin>\n'
        s += '    <DNSName>%s</DNSName>\n' % self.dns_name
        if self.origin_access_identity:
            val = get_oai_value(self.origin_access_identity)
            s += '    <OriginAccessIdentity>%s</OriginAccessIdentity>\n' % val
        s += '  </S3Origin>\n'
        return s