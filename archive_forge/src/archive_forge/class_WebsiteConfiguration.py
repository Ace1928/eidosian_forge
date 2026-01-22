class WebsiteConfiguration(object):
    """
    Website configuration for a bucket.

    :ivar suffix: Suffix that is appended to a request that is for a
        "directory" on the website endpoint (e.g. if the suffix is
        index.html and you make a request to samplebucket/images/
        the data that is returned will be for the object with the
        key name images/index.html).  The suffix must not be empty
        and must not include a slash character.

    :ivar error_key: The object key name to use when a 4xx class error
        occurs.  This key identifies the page that is returned when
        such an error occurs.

    :ivar redirect_all_requests_to: Describes the redirect behavior for every
        request to this bucket's website endpoint. If this value is non None,
        no other values are considered when configuring the website
        configuration for the bucket. This is an instance of
        ``RedirectLocation``.

    :ivar routing_rules: ``RoutingRules`` object which specifies conditions
        and redirects that apply when the conditions are met.

    """

    def __init__(self, suffix=None, error_key=None, redirect_all_requests_to=None, routing_rules=None):
        self.suffix = suffix
        self.error_key = error_key
        self.redirect_all_requests_to = redirect_all_requests_to
        if routing_rules is not None:
            self.routing_rules = routing_rules
        else:
            self.routing_rules = RoutingRules()

    def startElement(self, name, attrs, connection):
        if name == 'RoutingRules':
            self.routing_rules = RoutingRules()
            return self.routing_rules
        elif name == 'IndexDocument':
            return _XMLKeyValue([('Suffix', 'suffix')], container=self)
        elif name == 'ErrorDocument':
            return _XMLKeyValue([('Key', 'error_key')], container=self)

    def endElement(self, name, value, connection):
        pass

    def to_xml(self):
        parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<WebsiteConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">']
        if self.suffix is not None:
            parts.append(tag('IndexDocument', tag('Suffix', self.suffix)))
        if self.error_key is not None:
            parts.append(tag('ErrorDocument', tag('Key', self.error_key)))
        if self.redirect_all_requests_to is not None:
            parts.append(self.redirect_all_requests_to.to_xml())
        if self.routing_rules:
            parts.append(self.routing_rules.to_xml())
        parts.append('</WebsiteConfiguration>')
        return ''.join(parts)