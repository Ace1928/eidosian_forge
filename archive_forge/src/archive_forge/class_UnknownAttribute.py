from wsme.utils import _
class UnknownAttribute(ClientSideError):

    def __init__(self, fieldname, attributes, msg=''):
        self.fieldname = fieldname
        self.attributes = attributes
        self.msg = msg
        super(UnknownAttribute, self).__init__(self.msg)

    @property
    def faultstring(self):
        error = _('Unknown attribute for argument %(argn)s: %(attrs)s')
        if len(self.attributes) > 1:
            error = _('Unknown attributes for argument %(argn)s: %(attrs)s')
        str_attrs = ', '.join(self.attributes)
        return error % {'argn': self.fieldname, 'attrs': str_attrs}

    def add_fieldname(self, name):
        """Add a fieldname to concatenate the full name.

        Add a fieldname so that the whole hierarchy is displayed. Successive
        calls to this method will prepend ``name`` to the hierarchy of names.
        """
        if self.fieldname is not None:
            self.fieldname = '{}.{}'.format(name, self.fieldname)
        else:
            self.fieldname = name
        super(UnknownAttribute, self).__init__(self.msg)