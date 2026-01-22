class RoutingRules(list):

    def add_rule(self, rule):
        """

        :type rule: :class:`boto.s3.website.RoutingRule`
        :param rule: A routing rule.

        :return: This ``RoutingRules`` object is returned,
            so that it can chain subsequent calls.

        """
        self.append(rule)
        return self

    def startElement(self, name, attrs, connection):
        if name == 'RoutingRule':
            rule = RoutingRule(Condition(), Redirect())
            self.add_rule(rule)
            return rule

    def endElement(self, name, value, connection):
        pass

    def __repr__(self):
        return 'RoutingRules(%s)' % super(RoutingRules, self).__repr__()

    def to_xml(self):
        inner_text = []
        for rule in self:
            inner_text.append(rule.to_xml())
        return tag('RoutingRules', '\n'.join(inner_text))