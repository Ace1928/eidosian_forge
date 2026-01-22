class content_type(object):
    """This class represents a media type (aka a MIME content type), including parameters.

    You initialize these by passing in a content-type declaration
    string, such as "text/plain; charset=ascii", to the constructor or
    to the set() method.  If you provide no string value, the object
    returned will represent the wildcard */* content type.

    Normally you will get the value back by using str(), or optionally
    you can access the components via the 'major', 'minor', 'media_type',
    or 'parmdict' members.

    """

    def __init__(self, content_type_string=None, with_parameters=True):
        """Create a new content_type object.

        See the set() method for a description of the arguments.
        """
        if content_type_string:
            self.set(content_type_string, with_parameters=with_parameters)
        else:
            self.set('*/*')

    def set_parameters(self, parameter_list_or_dict):
        """Sets the optional paramters based upon the parameter list.

        The paramter list should be a semicolon-separated name=value string.
        Any paramters which already exist on this object will be deleted,
        unless they appear in the given paramter_list.

        """
        if isinstance(parameter_list_or_dict, dict):
            pl = parameter_list_or_dict
        else:
            pl, k = parse_parameter_list(parameter_list_or_dict)
            if k < len(parameter_list_or_dict):
                raise ParseError('Invalid parameter list', parameter_list_or_dict, k)
        self.parmdict = dict(pl)

    def set(self, content_type_string, with_parameters=True):
        """Parses the content type string and sets this object to it's value.

        For a more complete description of the arguments, see the
        documentation for the parse_media_type() function in this module.
        """
        mt, k = parse_media_type(content_type_string, with_parameters=with_parameters)
        if k < len(content_type_string):
            raise ParseError('Not a valid content type', content_type_string, k)
        major, minor, pdict = mt
        self._set_major(major)
        self._set_minor(minor)
        self.parmdict = dict(pdict)

    def _get_major(self):
        return self._major

    def _set_major(self, s):
        s = s.lower()
        if not is_token(s):
            raise ValueError('Major media type contains an invalid character')
        self._major = s

    def _get_minor(self):
        return self._minor

    def _set_minor(self, s):
        s = s.lower()
        if not is_token(s):
            raise ValueError('Minor media type contains an invalid character')
        self._minor = s
    major = property(_get_major, _set_major, doc='Major media classification')
    minor = property(_get_minor, _set_minor, doc='Minor media sub-classification')

    def __str__(self):
        """String value."""
        s = '%s/%s' % (self.major, self.minor)
        if self.parmdict:
            extra = '; '.join(['%s=%s' % (a[0], quote_string(a[1], False)) for a in self.parmdict.items()])
            s += '; ' + extra
        return s

    def __unicode__(self):
        """Unicode string value."""
        return str(self.__str__())

    def __repr__(self):
        """Python representation of this object."""
        s = '%s(%s)' % (self.__class__.__name__, repr(self.__str__()))
        return s

    def __hash__(self):
        """Hash this object; the hash is dependent only upon the value."""
        return hash(str(self))

    def __getstate__(self):
        """Pickler"""
        return str(self)

    def __setstate__(self, state):
        """Unpickler"""
        self.set(state)

    def __len__(self):
        """Logical length of this media type.
        For example:
           len('*/*')  -> 0
           len('image/*') -> 1
           len('image/png') -> 2
           len('text/plain; charset=utf-8')  -> 3
           len('text/plain; charset=utf-8; filename=xyz.txt') -> 4

        """
        if self.major == '*':
            return 0
        elif self.minor == '*':
            return 1
        else:
            return 2 + len(self.parmdict)

    def __eq__(self, other):
        """Equality test.

        Note that this is an exact match, including any parameters if any.
        """
        return self.major == other.major and self.minor == other.minor and (self.parmdict == other.parmdict)

    def __ne__(self, other):
        """Inequality test."""
        return not self.__eq__(other)

    def _get_media_type(self):
        """Returns the media 'type/subtype' string, without parameters."""
        return '%s/%s' % (self.major, self.minor)
    media_type = property(_get_media_type, doc="Returns the just the media type 'type/subtype' without any paramters (read-only).")

    def is_wildcard(self):
        """Returns True if this is a 'something/*' media type.
        """
        return self.minor == '*'

    def is_universal_wildcard(self):
        """Returns True if this is the unspecified '*/*' media type.
        """
        return self.major == '*' and self.minor == '*'

    def is_composite(self):
        """Is this media type composed of multiple parts.
        """
        return self.major == 'multipart' or self.major == 'message'

    def is_xml(self):
        """Returns True if this media type is XML-based.

        Note this does not consider text/html to be XML, but
        application/xhtml+xml is.
        """
        return self.minor == 'xml' or self.minor.endswith('+xml')