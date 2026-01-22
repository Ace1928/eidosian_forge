import xml.sax.saxutils
class FreeTextAnswer(object):
    template = '<FreeTextAnswer>%(items)s</FreeTextAnswer>'

    def __init__(self, default=None, constraints=None, num_lines=None):
        self.default = default
        if constraints is None:
            self.constraints = Constraints()
        else:
            self.constraints = Constraints(constraints)
        self.num_lines = num_lines

    def get_as_xml(self):
        items = [self.constraints]
        if self.default:
            items.append(SimpleField('DefaultText', self.default))
        if self.num_lines:
            items.append(NumberOfLinesSuggestion(self.num_lines))
        items = ''.join((item.get_as_xml() for item in items))
        return self.template % vars()