import xml.sax.saxutils
class LengthConstraint(Constraint):
    attribute_names = ('minLength', 'maxLength')
    template = '<Length %(attrs)s />'

    def __init__(self, min_length=None, max_length=None):
        self.attribute_values = (min_length, max_length)