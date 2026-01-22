import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
class SoupStrainer(object):
    """Encapsulates a number of ways of matching a markup element (tag or
    string).

    This is primarily used to underpin the find_* methods, but you can
    create one yourself and pass it in as `parse_only` to the
    `BeautifulSoup` constructor, to parse a subset of a large
    document.
    """

    def __init__(self, name=None, attrs={}, string=None, **kwargs):
        """Constructor.

        The SoupStrainer constructor takes the same arguments passed
        into the find_* methods. See the online documentation for
        detailed explanations.

        :param name: A filter on tag name.
        :param attrs: A dictionary of filters on attribute values.
        :param string: A filter for a NavigableString with specific text.
        :kwargs: A dictionary of filters on attribute values.
        """
        if string is None and 'text' in kwargs:
            string = kwargs.pop('text')
            warnings.warn("The 'text' argument to the SoupStrainer constructor is deprecated. Use 'string' instead.", DeprecationWarning, stacklevel=2)
        self.name = self._normalize_search_value(name)
        if not isinstance(attrs, dict):
            kwargs['class'] = attrs
            attrs = None
        if 'class_' in kwargs:
            kwargs['class'] = kwargs['class_']
            del kwargs['class_']
        if kwargs:
            if attrs:
                attrs = attrs.copy()
                attrs.update(kwargs)
            else:
                attrs = kwargs
        normalized_attrs = {}
        for key, value in list(attrs.items()):
            normalized_attrs[key] = self._normalize_search_value(value)
        self.attrs = normalized_attrs
        self.string = self._normalize_search_value(string)
        self.text = self.string

    def _normalize_search_value(self, value):
        if isinstance(value, str) or isinstance(value, Callable) or hasattr(value, 'match') or isinstance(value, bool) or (value is None):
            return value
        if isinstance(value, bytes):
            return value.decode('utf8')
        if hasattr(value, '__iter__'):
            new_value = []
            for v in value:
                if hasattr(v, '__iter__') and (not isinstance(v, bytes)) and (not isinstance(v, str)):
                    new_value.append(v)
                else:
                    new_value.append(self._normalize_search_value(v))
            return new_value
        return str(str(value))

    def __str__(self):
        """A human-readable representation of this SoupStrainer."""
        if self.string:
            return self.string
        else:
            return '%s|%s' % (self.name, self.attrs)

    def search_tag(self, markup_name=None, markup_attrs={}):
        """Check whether a Tag with the given name and attributes would
        match this SoupStrainer.

        Used prospectively to decide whether to even bother creating a Tag
        object.

        :param markup_name: A tag name as found in some markup.
        :param markup_attrs: A dictionary of attributes as found in some markup.

        :return: True if the prospective tag would match this SoupStrainer;
            False otherwise.
        """
        found = None
        markup = None
        if isinstance(markup_name, Tag):
            markup = markup_name
            markup_attrs = markup
        if isinstance(self.name, str):
            if markup and (not markup.prefix) and (self.name != markup.name):
                return False
        call_function_with_tag_data = isinstance(self.name, Callable) and (not isinstance(markup_name, Tag))
        if not self.name or call_function_with_tag_data or (markup and self._matches(markup, self.name)) or (not markup and self._matches(markup_name, self.name)):
            if call_function_with_tag_data:
                match = self.name(markup_name, markup_attrs)
            else:
                match = True
                markup_attr_map = None
                for attr, match_against in list(self.attrs.items()):
                    if not markup_attr_map:
                        if hasattr(markup_attrs, 'get'):
                            markup_attr_map = markup_attrs
                        else:
                            markup_attr_map = {}
                            for k, v in markup_attrs:
                                markup_attr_map[k] = v
                    attr_value = markup_attr_map.get(attr)
                    if not self._matches(attr_value, match_against):
                        match = False
                        break
            if match:
                if markup:
                    found = markup
                else:
                    found = markup_name
        if found and self.string and (not self._matches(found.string, self.string)):
            found = None
        return found
    searchTag = search_tag

    def search(self, markup):
        """Find all items in `markup` that match this SoupStrainer.

        Used by the core _find_all() method, which is ultimately
        called by all find_* methods.

        :param markup: A PageElement or a list of them.
        """
        found = None
        if hasattr(markup, '__iter__') and (not isinstance(markup, (Tag, str))):
            for element in markup:
                if isinstance(element, NavigableString) and self.search(element):
                    found = element
                    break
        elif isinstance(markup, Tag):
            if not self.string or self.name or self.attrs:
                found = self.search_tag(markup)
        elif isinstance(markup, NavigableString) or isinstance(markup, str):
            if not self.name and (not self.attrs) and self._matches(markup, self.string):
                found = markup
        else:
            raise Exception("I don't know how to match against a %s" % markup.__class__)
        return found

    def _matches(self, markup, match_against, already_tried=None):
        result = False
        if isinstance(markup, list) or isinstance(markup, tuple):
            for item in markup:
                if self._matches(item, match_against):
                    return True
            if self._matches(' '.join(markup), match_against):
                return True
            return False
        if match_against is True:
            return markup is not None
        if isinstance(match_against, Callable):
            return match_against(markup)
        original_markup = markup
        if isinstance(markup, Tag):
            markup = markup.name
        markup = self._normalize_search_value(markup)
        if markup is None:
            return not match_against
        if hasattr(match_against, '__iter__') and (not isinstance(match_against, str)):
            if not already_tried:
                already_tried = set()
            for item in match_against:
                if item.__hash__:
                    key = item
                else:
                    key = id(item)
                if key in already_tried:
                    continue
                else:
                    already_tried.add(key)
                    if self._matches(original_markup, item, already_tried):
                        return True
            else:
                return False
        match = False
        if not match and isinstance(match_against, str):
            match = markup == match_against
        if not match and hasattr(match_against, 'search'):
            return match_against.search(markup)
        if not match and isinstance(original_markup, Tag) and original_markup.prefix:
            return self._matches(original_markup.prefix + ':' + original_markup.name, match_against)
        return match