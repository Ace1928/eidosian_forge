import re
from formencode.rewritingparser import RewritingParser, html_quote
class FillingParser(RewritingParser):
    """
    Fills HTML with default values, as in a form.

    Examples::

        >>> defaults = dict(name='Bob Jones',
        ...             occupation='Crazy Cultist',
        ...             address='14 W. Canal\\nNew Guinea',
        ...             living='no',
        ...             nice_guy=0)
        >>> parser = FillingParser(defaults)
        >>> parser.feed('''<input type="text" name="name" value="fill">
        ... <select name="occupation"> <option value="">Default</option>
        ... <option value="Crazy Cultist">Crazy cultist</option> </select>
        ... <textarea cols="20" style="width: 100%" name="address">
        ... An address</textarea>
        ... <input type="radio" name="living" value="yes">
        ... <input type="radio" name="living" value="no">
        ... <input type="checkbox" name="nice_guy" checked="checked">''')
        >>> parser.close()
        >>> print (parser.text()) # doctest: +NORMALIZE_WHITESPACE
        <input type="text" name="name" value="Bob Jones">
        <select name="occupation">
        <option value="">Default</option>
        <option value="Crazy Cultist" selected="selected">Crazy cultist</option>
        </select>
        <textarea cols="20" style="width: 100%" name="address">14 W. Canal
        New Guinea</textarea>
        <input type="radio" name="living" value="yes">
        <input type="radio" name="living" value="no" checked="checked">
        <input type="checkbox" name="nice_guy">

    """
    default_encoding = 'utf8'
    text_input_types = set('text hidden search tel url email datetime date month week time datetime-local number range color'.split())

    def __init__(self, defaults, errors=None, use_all_keys=False, error_formatters=None, error_class='error', add_attributes=None, listener=None, auto_error_formatter=None, text_as_default=False, checkbox_checked_if_present=False, encoding=None, prefix_error=True, force_defaults=True, skip_passwords=False, data_formencode_form=None, data_formencode_ignore=None):
        RewritingParser.__init__(self)
        self.source = None
        self.lines = None
        self.source_pos = None
        self.defaults = defaults
        self.in_textarea = None
        self.skip_textarea = False
        self.last_textarea_name = None
        self.in_select = None
        self.skip_next = False
        self.errors = errors or {}
        if isinstance(self.errors, str):
            self.errors = {None: self.errors}
        self.in_error = None
        self.skip_error = False
        self.use_all_keys = use_all_keys
        self.used_keys = set()
        self.used_errors = set()
        if error_formatters is None:
            self.error_formatters = default_formatter_dict
        else:
            self.error_formatters = error_formatters
        self.error_class = error_class
        self.add_attributes = add_attributes or {}
        self.listener = listener
        self.auto_error_formatter = auto_error_formatter
        self.text_as_default = text_as_default
        self.checkbox_checked_if_present = checkbox_checked_if_present
        self.encoding = encoding
        self.prefix_error = prefix_error
        self.force_defaults = force_defaults
        self.skip_passwords = skip_passwords
        self.data_formencode_form = data_formencode_form
        self.data_formencode_ignore = data_formencode_ignore

    def str_compare(self, str1, str2):
        """
        Compare the two objects as strings (coercing to strings if necessary).
        Also uses encoding to compare the strings.
        """
        if not isinstance(str1, str):
            str1 = str(str1)
        if type(str1) is type(str2):
            return str1 == str2
        if isinstance(str1, str):
            str1 = str1.encode(self.encoding or self.default_encoding)
        else:
            str2 = str2.encode(self.encoding or self.default_encoding)
        return str1 == str2

    def close(self):
        self.handle_misc(None)
        RewritingParser.close(self)
        unused_errors = self.errors.copy()
        for key in self.used_errors:
            if key in unused_errors:
                del unused_errors[key]
        if self.auto_error_formatter:
            for key, value in unused_errors.items():
                error_message = self.auto_error_formatter(value)
                error_message = '<!-- for: %s -->\n%s' % (key, error_message)
                self.insert_at_marker(key, error_message)
            unused_errors = {}
        if self.use_all_keys:
            unused = self.defaults.copy()
            for key in self.used_keys:
                if key in unused:
                    del unused[key]
            assert not unused, 'These keys from defaults were not used in the form: %s' % ', '.join(unused)
            if unused_errors:
                error_text = ['%s: %s' % (key, self.errors[key]) for key in sorted(unused_errors)]
                assert False, 'These errors were not used in the form: %s' % ', '.join(error_text)
        if self.encoding is not None:
            new_content = []
            for item in self._content:
                new_content.append(item)
            self._content = new_content
        self._text = self._get_text()

    def skip_output(self):
        return self.in_textarea and self.skip_textarea or self.skip_error

    def add_key(self, key):
        self.used_keys.add(key)

    def handle_starttag(self, tag, attrs, startend=False):
        self.write_pos()
        if self.data_formencode_form:
            for a in attrs:
                if a[0] == 'data-formencode-form':
                    if a[1] != self.data_formencode_form:
                        return
        if self.data_formencode_ignore:
            for a in attrs:
                if a[0] == 'data-formencode-ignore':
                    return
        if tag == 'input':
            self.handle_input(attrs, startend)
        elif tag == 'textarea':
            self.handle_textarea(attrs)
        elif tag == 'select':
            self.handle_select(attrs)
        elif tag == 'option':
            self.handle_option(attrs)
            return
        elif tag == 'form:error':
            self.handle_error(attrs)
            return
        elif tag == 'form:iferror':
            self.handle_iferror(attrs)
            return
        else:
            return
        if self.listener:
            self.listener.listen_input(self, tag, attrs)

    def handle_endtag(self, tag):
        self.write_pos()
        if tag == 'textarea':
            self.handle_end_textarea()
        elif tag == 'select':
            self.handle_end_select()
        elif tag == 'form:error':
            self.handle_end_error()
        elif tag == 'form:iferror':
            self.handle_end_iferror()

    def handle_startendtag(self, tag, attrs):
        return self.handle_starttag(tag, attrs, True)

    def handle_iferror(self, attrs):
        name = self.get_attr(attrs, 'name')
        assert name, 'Name attribute in <iferror> required at %i:%i' % self.getpos()
        notted = name.startswith('not ')
        if notted:
            name = name.split(None, 1)[1]
        self.in_error = name
        ok = self.errors.get(name)
        if notted:
            ok = not ok
        if not ok:
            self.skip_error = True
        self.skip_next = True

    def handle_end_iferror(self):
        self.in_error = None
        self.skip_error = False
        self.skip_next = True

    def handle_error(self, attrs):
        name = self.get_attr(attrs, 'name')
        if name is None:
            name = self.in_error
        assert name is not None, 'Name attribute in <form:error> required if not contained in <form:iferror> at %i:%i' % self.getpos()
        formatter = self.get_attr(attrs, 'format') or 'default'
        error = self.errors.get(name, '')
        if error:
            error = self.error_formatters[formatter](error)
            self.write_text(error)
        self.skip_next = True
        self.used_errors.add(name)

    def handle_end_error(self):
        self.skip_next = True

    def handle_input(self, attrs, startend):
        t = (self.get_attr(attrs, 'type') or 'text').lower()
        name = self.get_attr(attrs, 'name')
        if self.prefix_error:
            self.write_marker(name)
        value = self.defaults.get(name)
        if name in self.add_attributes:
            for attr_name, attr_value in self.add_attributes[name].items():
                if attr_name.startswith('+'):
                    attr_name = attr_name[1:]
                    self.set_attr(attrs, attr_name, self.get_attr(attrs, attr_name, '') + attr_value)
                else:
                    self.set_attr(attrs, attr_name, attr_value)
        if self.error_class and self.errors.get(self.get_attr(attrs, 'name')):
            self.add_class(attrs, self.error_class)
        if t in self.text_input_types:
            if value is None and (not self.force_defaults):
                value = self.get_attr(attrs, 'value', '')
            self.set_attr(attrs, 'value', value)
            self.write_tag('input', attrs, startend)
            self.skip_next = True
            self.add_key(name)
        elif t == 'checkbox':
            if self.force_defaults:
                selected = False
            else:
                selected = self.get_attr(attrs, 'checked')
            if not self.get_attr(attrs, 'value'):
                if self.checkbox_checked_if_present:
                    selected = name in self.defaults
                else:
                    selected = value
            elif self.selected_multiple(value, self.get_attr(attrs, 'value', '')):
                selected = True
            if selected:
                self.set_attr(attrs, 'checked', 'checked')
            else:
                self.del_attr(attrs, 'checked')
            self.write_tag('input', attrs, startend)
            self.skip_next = True
            self.add_key(name)
        elif t == 'radio':
            if self.str_compare(value, self.get_attr(attrs, 'value', '')):
                self.set_attr(attrs, 'checked', 'checked')
            elif self.force_defaults or name in self.defaults:
                self.del_attr(attrs, 'checked')
            self.write_tag('input', attrs, startend)
            self.skip_next = True
            self.add_key(name)
        elif t == 'password':
            if self.skip_passwords:
                return
            if value is None and (not self.force_defaults):
                value = value or self.get_attr(attrs, 'value', '')
            self.set_attr(attrs, 'value', value)
            self.write_tag('input', attrs, startend)
            self.skip_next = True
            self.add_key(name)
        elif t in ('file', 'image'):
            self.write_tag('input', attrs, startend)
            self.skip_next = True
            self.add_key(name)
        elif t in ('submit', 'reset', 'button'):
            self.set_attr(attrs, 'value', value or self.get_attr(attrs, 'value', ''))
            self.write_tag('input', attrs, startend)
            self.skip_next = True
            self.add_key(name)
        elif self.text_as_default:
            if value is None:
                value = self.get_attr(attrs, 'value', '')
            self.set_attr(attrs, 'value', value)
            self.write_tag('input', attrs, startend)
            self.skip_next = True
            self.add_key(name)
        else:
            assert False, "I don't know about this kind of <input>: %s at %i:%i" % ((t,) + self.getpos())
        if not self.prefix_error:
            self.write_marker(name)

    def handle_textarea(self, attrs):
        name = self.get_attr(attrs, 'name')
        if self.prefix_error:
            self.write_marker(name)
        if self.error_class and self.errors.get(name):
            self.add_class(attrs, self.error_class)
        value = self.defaults.get(name, '')
        if value or self.force_defaults:
            self.write_tag('textarea', attrs)
            self.write_text(html_quote(value))
            self.write_text('</textarea>')
            self.skip_textarea = True
        self.in_textarea = True
        self.last_textarea_name = name
        self.add_key(name)

    def handle_end_textarea(self):
        if self.skip_textarea:
            self.skip_textarea = False
        else:
            self.write_text('</textarea>')
        self.in_textarea = False
        self.skip_next = True
        if not self.prefix_error:
            self.write_marker(self.last_textarea_name)
        self.last_textarea_name = None

    def handle_select(self, attrs):
        name = self.get_attr(attrs, 'name', False)
        if name and self.prefix_error:
            self.write_marker(name)
        if self.error_class and self.errors.get(name):
            self.add_class(attrs, self.error_class)
        self.in_select = self.get_attr(attrs, 'name', False)
        self.write_tag('select', attrs)
        self.skip_next = True
        self.add_key(self.in_select)

    def handle_end_select(self):
        self.write_text('</select>')
        self.skip_next = True
        if not self.prefix_error and self.in_select:
            self.write_marker(self.in_select)
        self.in_select = None

    def handle_option(self, attrs):
        assert self.in_select is not None, '<option> outside of <select> at %i:%i' % self.getpos()
        if self.in_select is not False:
            if self.force_defaults or self.in_select in self.defaults:
                if self.selected_multiple(self.defaults.get(self.in_select), self.get_attr(attrs, 'value', '')):
                    self.set_attr(attrs, 'selected', 'selected')
                    self.add_key(self.in_select)
                else:
                    self.del_attr(attrs, 'selected')
        self.write_tag('option', attrs)
        self.skip_next = True

    def selected_multiple(self, obj, value):
        """
        Returns true/false if obj indicates that value should be
        selected.  If obj has a __contains__ method it is used, otherwise
        identity is used.
        """
        if obj is None:
            return False
        if isinstance(obj, str):
            return obj == value
        if hasattr(obj, '__contains__'):
            if value in obj:
                return True
        if hasattr(obj, '__iter__'):
            for inner in obj:
                if self.str_compare(inner, value):
                    return True
        return self.str_compare(obj, value)

    def write_marker(self, marker):
        self._content.append((marker,))

    def insert_at_marker(self, marker, text):
        for i, item in enumerate(self._content):
            if item == (marker,):
                self._content.insert(i, text)
                break
        else:
            self._content.insert(0, text)