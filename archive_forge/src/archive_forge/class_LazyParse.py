from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
import six
class LazyParse(object):
    """Class provided when accessing a concept to lazily parse from args."""

    def __init__(self, parse, arg_getter):
        self.parse = parse
        self.arg_getter = arg_getter

    def Parse(self):
        try:
            return self.parse(self.arg_getter())
        except concepts.InitializationError as e:
            if required:
                raise ParseError(name, six.text_type(e))
            return None