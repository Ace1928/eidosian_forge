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
class RepresentationDefinition(WADLResolvableDefinition, HasParametersMixin):
    """A definition of the structure of a representation."""

    def __init__(self, application, resource, representation_tag):
        super(RepresentationDefinition, self).__init__(application)
        self.resource = resource
        self.tag = representation_tag

    def params(self, resource):
        return super(RepresentationDefinition, self).params(['query', 'plain'], resource)

    def parameter_names(self, resource):
        """Return the names of all parameters."""
        return [param.name for param in self.params(resource)]

    @property
    def media_type(self):
        """The media type of the representation described here."""
        return self.resolve_definition().tag.attrib['mediaType']

    def _make_boundary(self, all_parts):
        """Make a random boundary that does not appear in `all_parts`."""
        _width = len(repr(sys.maxsize - 1))
        _fmt = '%%0%dd' % _width
        token = random.randrange(sys.maxsize)
        boundary = '=' * 15 + _fmt % token + '=='
        if all_parts is None:
            return boundary
        b = boundary
        counter = 0
        while True:
            pattern = ('^--' + re.escape(b) + '(--)?$').encode('ascii')
            if not re.search(pattern, all_parts, flags=re.MULTILINE):
                break
            b = boundary + '.' + str(counter)
            counter += 1
        return b

    def _write_headers(self, buf, headers):
        """Write MIME headers to a file object."""
        for key, value in headers:
            buf.write(key.encode('UTF-8'))
            buf.write(b': ')
            buf.write(value.encode('UTF-8'))
            buf.write(b'\r\n')
        buf.write(b'\r\n')

    def _write_boundary(self, buf, boundary, closing=False):
        """Write a multipart boundary to a file object."""
        buf.write(b'--')
        buf.write(boundary.encode('UTF-8'))
        if closing:
            buf.write(b'--')
        buf.write(b'\r\n')

    def _generate_multipart_form(self, parts):
        """Generate a multipart/form-data message.

        This is very loosely based on the email module in the Python standard
        library.  However, that module doesn't really support directly embedding
        binary data in a form: various versions of Python have mangled line
        separators in different ways, and none of them get it quite right.
        Since we only need a tiny subset of MIME here, it's easier to implement
        it ourselves.

        :return: a tuple of two elements: the Content-Type of the message, and
            the entire encoded message as a byte string.
        """
        encoded_parts = []
        for is_binary, name, value in parts:
            buf = io.BytesIO()
            if is_binary:
                ctype = 'application/octet-stream'
                cdisp = 'form-data; name="%s"; filename="%s"' % (quote(name), quote(name))
            else:
                ctype = 'text/plain; charset="utf-8"'
                cdisp = 'form-data; name="%s"' % quote(name)
            self._write_headers(buf, [('MIME-Version', '1.0'), ('Content-Type', ctype), ('Content-Disposition', cdisp)])
            if is_binary:
                if not isinstance(value, bytes):
                    raise TypeError('bytes payload expected: %s' % type(value))
                buf.write(value)
            else:
                if not isinstance(value, _string_types):
                    raise TypeError('string payload expected: %s' % type(value))
                lines = re.split('\\r\\n|\\r|\\n', value)
                for line in lines[:-1]:
                    buf.write(line.encode('UTF-8'))
                    buf.write(b'\r\n')
                buf.write(lines[-1].encode('UTF-8'))
            encoded_parts.append(buf.getvalue())
        boundary = self._make_boundary(b'\r\n'.join(encoded_parts))
        buf = io.BytesIO()
        ctype = 'multipart/form-data; boundary="%s"' % quote(boundary)
        self._write_headers(buf, [('MIME-Version', '1.0'), ('Content-Type', ctype)])
        for encoded_part in encoded_parts:
            self._write_boundary(buf, boundary)
            buf.write(encoded_part)
            buf.write(b'\r\n')
        self._write_boundary(buf, boundary, closing=True)
        return (ctype, buf.getvalue())

    def bind(self, param_values, **kw_param_values):
        """Bind the definition to parameter values, creating a document.

        :return: A 2-tuple (media_type, document).
        """
        definition = self.resolve_definition()
        params = definition.params(self.resource)
        validated_values = self.validate_param_values(params, param_values, **kw_param_values)
        media_type = self.media_type
        if media_type == 'application/x-www-form-urlencoded':
            doc = urlencode(sorted(validated_values.items()))
        elif media_type == 'multipart/form-data':
            parts = []
            missing = object()
            for param in params:
                value = validated_values.get(param.name, missing)
                if value is not missing:
                    parts.append((param.type == 'binary', param.name, value))
            media_type, doc = self._generate_multipart_form(parts)
        elif media_type == 'application/json':
            doc = json.dumps(validated_values)
        else:
            raise ValueError("Unsupported media type: '%s'" % media_type)
        return (media_type, doc)

    def _definition_factory(self, id):
        """Turn a representation ID into a RepresentationDefinition."""
        return self.application.representation_definitions.get(id)

    def _get_definition_url(self):
        """Find the URL containing the representation's 'real' definition.

        If a representation's structure is defined by reference, the
        <representation> tag's 'href' attribute will contain the URL
        to the <representation> that defines the structure.
        """
        return self.tag.attrib.get('href')