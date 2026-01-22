import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
class Deb822ParsedTokenList(Generic[VE, ST], _Deb822ParsedTokenList_ContextManager['Deb822ParsedTokenList[VE, ST]']):

    def __init__(self, kvpair_element, interpreted_value_element, vtype, stype, str2value_parser, default_separator_factory, render):
        self._kvpair_element = kvpair_element
        self._token_list = LinkedList(interpreted_value_element)
        self._vtype = vtype
        self._stype = stype
        self._str2value_parser = str2value_parser
        self._default_separator_factory = default_separator_factory
        self._value_factory = _parser_to_value_factory(str2value_parser, vtype)
        self._render = render
        self._format_preserve_original_formatting = True
        self._formatter = one_value_per_line_trailing_separator
        self._changed = False
        self.__continuation_line_char = None
        assert self._token_list
        last_token = self._token_list.tail
        if last_token is not None and isinstance(last_token, Deb822NewlineAfterValueToken):
            self._token_list.pop()

    def __iter__(self):
        yield from (self._render(v) for v in self.value_parts)

    def __bool__(self):
        return next(iter(self), None) is not None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self._changed:
            self._update_field()
        return super().__exit__(exc_type, exc_val, exc_tb)

    @property
    def value_parts(self):
        yield from (v for v in self._token_list if isinstance(v, self._vtype))

    def _mark_changed(self):
        self._changed = True

    def iter_value_references(self):
        """Iterate over all values in the list (as ValueReferences)

        This is useful for doing inplace modification of the values or even
        streaming removal of field values.  It is in general also more
        efficient when more than one value is updated or removed.
        """
        yield from (ValueReference(cast('LinkedListNode[VE]', n), self._render, self._value_factory, self._remove_node, self._mark_changed) for n in self._token_list.iter_nodes() if isinstance(n.value, self._vtype))

    def append_separator(self, space_after_separator=True):
        separator_token = self._default_separator_factory()
        if separator_token.is_whitespace:
            space_after_separator = False
        self._changed = True
        self._append_continuation_line_token_if_necessary()
        self._token_list.append(separator_token)
        if space_after_separator and (not separator_token.is_whitespace):
            self._token_list.append(Deb822WhitespaceToken(' '))

    def replace(self, orig_value, new_value):
        """Replace the first instance of a value with another

        This method will *not* affect the validity of ValueReferences.
        """
        vtype = self._vtype
        for node in self._token_list.iter_nodes():
            if isinstance(node.value, vtype) and self._render(node.value) == orig_value:
                node.value = self._value_factory(new_value)
                self._changed = True
                break
        else:
            raise ValueError('list.replace(x, y): x not in list')

    def remove(self, value):
        """Remove the first instance of a value

        Removal will invalidate ValueReferences to the value being removed.
        ValueReferences to other values will be unaffected.
        """
        vtype = self._vtype
        for node in self._token_list.iter_nodes():
            if isinstance(node.value, vtype) and self._render(node.value) == value:
                node_to_remove = node
                break
        else:
            raise ValueError('list.remove(x): x not in list')
        return self._remove_node(node_to_remove)

    def _remove_node(self, node_to_remove):
        vtype = self._vtype
        self._changed = True
        first_value_on_lhs = None
        first_value_on_rhs = None
        comment_before_previous_value = False
        comment_before_next_value = False
        for past_node in node_to_remove.iter_previous(skip_current=True):
            past_token = past_node.value
            if isinstance(past_token, Deb822Token) and past_token.is_comment:
                comment_before_previous_value = True
                continue
            if isinstance(past_token, vtype):
                first_value_on_lhs = past_node
                break
        for future_node in node_to_remove.iter_next(skip_current=True):
            future_token = future_node.value
            if isinstance(future_token, Deb822Token) and future_token.is_comment:
                comment_before_next_value = True
                continue
            if isinstance(future_token, vtype):
                first_value_on_rhs = future_node
                break
        if first_value_on_rhs is None and first_value_on_lhs is None:
            self._token_list.clear()
            return
        if first_value_on_lhs is not None and (not comment_before_previous_value):
            delete_lhs_of_node = True
        elif first_value_on_rhs is not None and (not comment_before_next_value):
            delete_lhs_of_node = False
        else:
            delete_lhs_of_node = first_value_on_lhs is not None
        if delete_lhs_of_node:
            first_remain_lhs = first_value_on_lhs
            first_remain_rhs = node_to_remove.next_node
        else:
            first_remain_lhs = node_to_remove.previous_node
            first_remain_rhs = first_value_on_rhs
        if first_remain_lhs is None:
            self._token_list.head_node = first_remain_rhs
        if first_remain_rhs is None:
            self._token_list.tail_node = first_remain_lhs
        LinkedListNode.link_nodes(first_remain_lhs, first_remain_rhs)

    def append(self, value):
        vt = self._value_factory(value)
        self.append_value(vt)

    def append_value(self, vt):
        value_parts = self._token_list
        if value_parts:
            needs_separator = False
            stype = self._stype
            vtype = self._vtype
            for t in reversed(value_parts):
                if isinstance(t, vtype):
                    needs_separator = True
                    break
                if isinstance(t, stype):
                    break
            if needs_separator:
                self.append_separator()
        else:
            self._token_list.append(Deb822WhitespaceToken(' '))
        self._append_continuation_line_token_if_necessary()
        self._changed = True
        value_parts.append(vt)

    def _previous_is_newline(self):
        tail = self._token_list.tail
        return tail is not None and tail.convert_to_text().endswith('\n')

    def append_newline(self):
        if self._previous_is_newline():
            raise ValueError('Cannot add a newline after a token that ends on a newline')
        self._token_list.append(Deb822NewlineAfterValueToken())

    def append_comment(self, comment_text):
        tail = self._token_list.tail
        if tail is None or not tail.convert_to_text().endswith('\n'):
            self.append_newline()
        comment_token = Deb822CommentToken(_format_comment(comment_text))
        self._token_list.append(comment_token)

    @property
    def _continuation_line_char(self):
        char = self.__continuation_line_char
        if char is None:
            char = ' '
            for token in self._token_list:
                if isinstance(token, Deb822ValueContinuationToken):
                    char = token.text
                    break
            self.__continuation_line_char = char
        return char

    def _append_continuation_line_token_if_necessary(self):
        tail = self._token_list.tail
        if tail is not None and tail.convert_to_text().endswith('\n'):
            self._token_list.append(Deb822ValueContinuationToken(self._continuation_line_char))

    def reformat_when_finished(self):
        self._enable_reformatting()
        self._changed = True

    def _enable_reformatting(self):
        self._format_preserve_original_formatting = False

    def no_reformatting_when_finished(self):
        self._format_preserve_original_formatting = True

    def value_formatter(self, formatter, force_reformat=False):
        """Use a custom formatter when formatting the value

        :param formatter: A formatter (see debian._deb822_repro.formatter.format_field
          for details)
        :param force_reformat: If True, always reformat the field even if there are
          no (other) changes performed.  By default, fields are only reformatted if
          they are changed.
        """
        self._formatter = formatter
        self._format_preserve_original_formatting = False
        if force_reformat:
            self._changed = True

    def clear(self):
        """Like list.clear() - removes all content (including comments and spaces)"""
        if self._token_list:
            self._changed = True
        self._token_list.clear()

    def _iter_content_as_tokens(self):
        for te in self._token_list:
            if isinstance(te, Deb822Element):
                yield from te.iter_tokens()
            else:
                yield te

    def _generate_reformatted_field_content(self):
        separator_token = self._default_separator_factory()
        vtype = self._vtype
        stype = self._stype
        token_list = self._token_list

        def _token_iter():
            text = ''
            for te in token_list:
                if isinstance(te, Deb822Token):
                    if te.is_comment:
                        yield FormatterContentToken.comment_token(te.text)
                    elif isinstance(te, stype):
                        text = te.text
                        yield FormatterContentToken.separator_token(text)
                else:
                    assert isinstance(te, vtype)
                    text = te.convert_to_text()
                    yield FormatterContentToken.value_token(text)
        return format_field(self._formatter, self._kvpair_element.field_name, FormatterContentToken.separator_token(separator_token.text), _token_iter())

    def _generate_field_content(self):
        return ''.join((t.text for t in self._iter_content_as_tokens()))

    def _update_field(self):
        kvpair_element = self._kvpair_element
        field_name = kvpair_element.field_name
        token_list = self._token_list
        tail = token_list.tail
        had_tokens = False
        for t in self._iter_content_as_tokens():
            had_tokens = True
            if not t.is_comment and (not t.is_whitespace):
                break
        else:
            if had_tokens:
                raise ValueError('Field must be completely empty or have content (i.e. non-whitespace and non-comments)')
        if tail is not None:
            if isinstance(tail, Deb822Token) and tail.is_comment:
                raise ValueError('Fields must not end on a comment')
            if not tail.convert_to_text().endswith('\n'):
                self.append_newline()
            if self._format_preserve_original_formatting:
                value_text = self._generate_field_content()
                text = ':'.join((field_name, value_text))
            else:
                text = self._generate_reformatted_field_content()
            new_content = text.splitlines(keepends=True)
        else:
            new_content = [field_name + ':\n']
        deb822_file = parse_deb822_file(iter(new_content))
        error_token = deb822_file.find_first_error_element()
        if error_token:
            raise ValueError('Syntax error in new field value for ' + field_name)
        paragraph = next(iter(deb822_file))
        assert isinstance(paragraph, Deb822NoDuplicateFieldsParagraphElement)
        new_kvpair_element = paragraph.get_kvpair_element(field_name)
        assert new_kvpair_element is not None
        kvpair_element.value_element = new_kvpair_element.value_element
        self._changed = False

    def sort_elements(self, *, key=None, reverse=False):
        """Sort the elements (abstract values) in this list.

        This method will sort the logical values of the list. It will
        attempt to preserve comments associated with a given value where
        possible.  Whether space and separators are preserved depends on
        the contents of the field as well as the formatting settings.

        Sorting (without reformatting) is likely to leave you with "awkward"
        whitespace. Therefore, you almost always want to apply reformatting
        such as the reformat_when_finished() method.

        Sorting will invalidate all ValueReferences.
        """
        comment_start_node = None
        vtype = self._vtype
        stype = self._stype

        def key_func(x):
            if key:
                return key(x[0])
            return x[0].convert_to_text()
        parts = []
        for node in self._token_list.iter_nodes():
            value = node.value
            if isinstance(value, Deb822Token) and value.is_comment:
                if comment_start_node is None:
                    comment_start_node = node
                continue
            if isinstance(value, vtype):
                comments = []
                if comment_start_node is not None:
                    for keep_node in comment_start_node.iter_next(skip_current=False):
                        if keep_node is node:
                            break
                        comments.append(keep_node.value)
                parts.append((value, comments))
                comment_start_node = None
        parts.sort(key=key_func, reverse=reverse)
        self._changed = True
        self._token_list.clear()
        first_value = True
        separator_is_space = self._default_separator_factory().is_whitespace
        for value, comments in parts:
            if first_value:
                first_value = False
                if comments:
                    comments = [x for x in comments if not isinstance(x, stype)]
                    self.append_newline()
            else:
                if not separator_is_space and (not any((isinstance(x, stype) for x in comments))):
                    self.append_separator(space_after_separator=False)
                if comments:
                    self.append_newline()
                else:
                    self._token_list.append(Deb822WhitespaceToken(' '))
            self._token_list.extend(comments)
            self.append_value(value)

    def sort(self, *, key=None, **kwargs):
        """Sort the values (rendered as str) in this list.

        This method will sort the logical values of the list. It will
        attempt to preserve comments associated with a given value where
        possible.  Whether space and separators are preserved depends on
        the contents of the field as well as the formatting settings.

        Sorting (without reformatting) is likely to leave you with "awkward"
        whitespace. Therefore, you almost always want to apply reformatting
        such as the reformat_when_finished() method.

        Sorting will invalidate all ValueReferences.
        """
        if key is not None:
            render = self._render
            kwargs['key'] = lambda vt: key(render(vt))
        self.sort_elements(**kwargs)