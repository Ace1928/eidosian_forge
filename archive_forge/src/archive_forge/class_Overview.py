import xml.sax.saxutils
class Overview(OrderedContent):
    template = '<Overview>%(content)s</Overview>'

    def get_as_params(self, label='Overview'):
        return {label: self.get_as_xml()}

    def get_as_xml(self):
        content = super(Overview, self).get_as_xml()
        return self.template % vars()