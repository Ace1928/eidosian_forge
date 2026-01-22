import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
class ResponseDefinition(HasParametersMixin):
    """A wrapper around the description of a response to a method."""

    def __init__(self, resource, response_tag, headers=None):
        """Initialize with a <response> tag.

        :param response_tag: An ElementTree <response> tag.
        """
        self.application = resource.application
        self.resource = resource
        self.tag = response_tag
        self.headers = headers

    def __iter__(self):
        """Get an iterator over the representation definitions.

        These are the representations returned in response to an
        invocation of this method.
        """
        path = wadl_xpath('representation')
        for representation_tag in self.tag.findall(path):
            yield RepresentationDefinition(self.resource.application, self.resource, representation_tag)

    def bind(self, headers):
        """Bind the response to a set of HTTP headers.

        A WADL response can have associated header parameters, but no
        other kind.
        """
        return ResponseDefinition(self.resource, self.tag, headers)

    def get_parameter(self, param_name):
        """Find a header parameter within the response."""
        for param_tag in self.tag.findall(wadl_xpath('param')):
            if param_tag.attrib.get('name') == param_name and param_tag.attrib.get('style') == 'header':
                return Parameter(self, param_tag)
        return None

    def get_parameter_value(self, parameter):
        """Find the value of a parameter, given the Parameter object."""
        if self.headers is None:
            raise NoBoundRepresentationError('Response object is not bound to any headers.')
        if parameter.style != 'header':
            raise NotImplementedError("Don't know how to find value for a parameter of type %s." % parameter.style)
        return self.headers.get(parameter.name)

    def get_representation_definition(self, media_type):
        """Get one of the possible representations of the response."""
        if self.tag is None:
            return None
        for representation in self:
            if representation.media_type == media_type:
                return representation
        return None