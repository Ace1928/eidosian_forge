from calendar import timegm
from decimal import Decimal as MyDecimal, ROUND_HALF_EVEN
from email.utils import formatdate
import six
from flask_restful import marshal
from flask import url_for, request
class FormattedString(Raw):
    """
    FormattedString is used to interpolate other values from
    the response into this field. The syntax for the source string is
    the same as the string :meth:`~str.format` method from the python
    stdlib.

    Ex::

        fields = {
            'name': fields.String,
            'greeting': fields.FormattedString("Hello {name}")
        }
        data = {
            'name': 'Doug',
        }
        marshal(data, fields)
    """

    def __init__(self, src_str):
        """
        :param string src_str: the string to format with the other
        values from the response.
        """
        super(FormattedString, self).__init__()
        self.src_str = six.text_type(src_str)

    def output(self, key, obj):
        try:
            data = to_marshallable_type(obj)
            return self.src_str.format(**data)
        except (TypeError, IndexError) as error:
            raise MarshallingException(error)