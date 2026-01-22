import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
class NavigableString(str, PageElement):
    """A Python Unicode string that is part of a parse tree.

    When Beautiful Soup parses the markup <b>penguin</b>, it will
    create a NavigableString for the string "penguin".
    """
    PREFIX = ''
    SUFFIX = ''

    def __new__(cls, value):
        """Create a new NavigableString.

        When unpickling a NavigableString, this method is called with
        the string in DEFAULT_OUTPUT_ENCODING. That encoding needs to be
        passed in to the superclass's __new__ or the superclass won't know
        how to handle non-ASCII characters.
        """
        if isinstance(value, str):
            u = str.__new__(cls, value)
        else:
            u = str.__new__(cls, value, DEFAULT_OUTPUT_ENCODING)
        u.setup()
        return u

    def __deepcopy__(self, memo, recursive=False):
        """A copy of a NavigableString has the same contents and class
        as the original, but it is not connected to the parse tree.

        :param recursive: This parameter is ignored; it's only defined
           so that NavigableString.__deepcopy__ implements the same
           signature as Tag.__deepcopy__.
        """
        return type(self)(self)

    def __copy__(self):
        """A copy of a NavigableString can only be a deep copy, because
        only one PageElement can occupy a given place in a parse tree.
        """
        return self.__deepcopy__({})

    def __getnewargs__(self):
        return (str(self),)

    def __getattr__(self, attr):
        """text.string gives you text. This is for backwards
        compatibility for Navigable*String, but for CData* it lets you
        get the string without the CData wrapper."""
        if attr == 'string':
            return self
        else:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, attr))

    def output_ready(self, formatter='minimal'):
        """Run the string through the provided formatter.

        :param formatter: A Formatter object, or a string naming one of the standard formatters.
        """
        output = self.format_string(self, formatter)
        return self.PREFIX + output + self.SUFFIX

    @property
    def name(self):
        """Since a NavigableString is not a Tag, it has no .name.

        This property is implemented so that code like this doesn't crash
        when run on a mixture of Tag and NavigableString objects:
            [x.name for x in tag.children]
        """
        return None

    @name.setter
    def name(self, name):
        """Prevent NavigableString.name from ever being set."""
        raise AttributeError('A NavigableString cannot be given a name.')

    def _all_strings(self, strip=False, types=PageElement.default):
        """Yield all strings of certain classes, possibly stripping them.

        This makes it easy for NavigableString to implement methods
        like get_text() as conveniences, creating a consistent
        text-extraction API across all PageElements.

        :param strip: If True, all strings will be stripped before being
            yielded.

        :param types: A tuple of NavigableString subclasses. If this
            NavigableString isn't one of those subclasses, the
            sequence will be empty. By default, the subclasses
            considered are NavigableString and CData objects. That
            means no comments, processing instructions, etc.

        :yield: A sequence that either contains this string, or is empty.

        """
        if types is self.default:
            types = Tag.DEFAULT_INTERESTING_STRING_TYPES
        my_type = type(self)
        if types is not None:
            if isinstance(types, type):
                if my_type is not types:
                    return
            elif my_type not in types:
                return
        value = self
        if strip:
            value = value.strip()
        if len(value) > 0:
            yield value
    strings = property(_all_strings)