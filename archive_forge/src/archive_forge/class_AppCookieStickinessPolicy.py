from boto.resultset import ResultSet
class AppCookieStickinessPolicy(object):

    def __init__(self, connection=None):
        self.cookie_name = None
        self.policy_name = None

    def __repr__(self):
        return 'AppCookieStickiness(%s, %s)' % (self.policy_name, self.cookie_name)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'CookieName':
            self.cookie_name = value
        elif name == 'PolicyName':
            self.policy_name = value