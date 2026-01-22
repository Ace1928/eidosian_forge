from __future__ import unicode_literals
import re
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.filters.base import Filter
from tensorboard._vendor.html5lib.filters.sanitizer import allowed_protocols
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach import callbacks as linkify_callbacks
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
class LinkifyFilter(Filter):
    """html5lib filter that linkifies text

    This will do the following:

    * convert email addresses into links
    * convert urls into links
    * edit existing links by running them through callbacks--the default is to
      add a ``rel="nofollow"``

    This filter can be used anywhere html5lib filters can be used.

    """

    def __init__(self, source, callbacks=None, skip_tags=None, parse_email=False, url_re=URL_RE, email_re=EMAIL_RE):
        """Creates a LinkifyFilter instance

        :arg TreeWalker source: stream

        :arg list callbacks: list of callbacks to run when adjusting tag attributes;
            defaults to ``bleach.linkifier.DEFAULT_CALLBACKS``

        :arg list skip_tags: list of tags that you don't want to linkify the
            contents of; for example, you could set this to ``['pre']`` to skip
            linkifying contents of ``pre`` tags

        :arg bool parse_email: whether or not to linkify email addresses

        :arg re url_re: url matching regex

        :arg re email_re: email matching regex

        """
        super(LinkifyFilter, self).__init__(source)
        self.callbacks = callbacks or []
        self.skip_tags = skip_tags or []
        self.parse_email = parse_email
        self.url_re = url_re
        self.email_re = email_re

    def apply_callbacks(self, attrs, is_new):
        """Given an attrs dict and an is_new bool, runs through callbacks

        Callbacks can return an adjusted attrs dict or ``None``. In the case of
        ``None``, we stop going through callbacks and return that and the link
        gets dropped.

        :arg dict attrs: map of ``(namespace, name)`` -> ``value``

        :arg bool is_new: whether or not this link was added by linkify

        :returns: adjusted attrs dict or ``None``

        """
        for cb in self.callbacks:
            attrs = cb(attrs, is_new)
            if attrs is None:
                return None
        return attrs

    def extract_character_data(self, token_list):
        """Extracts and squashes character sequences in a token stream"""
        out = []
        for token in token_list:
            token_type = token['type']
            if token_type in ['Characters', 'SpaceCharacters']:
                out.append(token['data'])
        return u''.join(out)

    def handle_email_addresses(self, src_iter):
        """Handle email addresses in character tokens"""
        for token in src_iter:
            if token['type'] == 'Characters':
                text = token['data']
                new_tokens = []
                end = 0
                for match in self.email_re.finditer(text):
                    if match.start() > end:
                        new_tokens.append({u'type': u'Characters', u'data': text[end:match.start()]})
                    attrs = {(None, u'href'): u'mailto:%s' % match.group(0), u'_text': match.group(0)}
                    attrs = self.apply_callbacks(attrs, True)
                    if attrs is None:
                        new_tokens.append({u'type': u'Characters', u'data': match.group(0)})
                    else:
                        _text = attrs.pop(u'_text', '')
                        attrs = alphabetize_attributes(attrs)
                        new_tokens.extend([{u'type': u'StartTag', u'name': u'a', u'data': attrs}, {u'type': u'Characters', u'data': force_unicode(_text)}, {u'type': u'EndTag', u'name': 'a'}])
                    end = match.end()
                if new_tokens:
                    if end < len(text):
                        new_tokens.append({u'type': u'Characters', u'data': text[end:]})
                    for new_token in new_tokens:
                        yield new_token
                    continue
            yield token

    def strip_non_url_bits(self, fragment):
        """Strips non-url bits from the url

        This accounts for over-eager matching by the regex.

        """
        prefix = suffix = ''
        while fragment:
            if fragment.startswith(u'('):
                prefix = prefix + u'('
                fragment = fragment[1:]
                if fragment.endswith(u')'):
                    suffix = u')' + suffix
                    fragment = fragment[:-1]
                continue
            if fragment.endswith(u')') and u'(' not in fragment:
                fragment = fragment[:-1]
                suffix = u')' + suffix
                continue
            if fragment.endswith(u','):
                fragment = fragment[:-1]
                suffix = u',' + suffix
                continue
            if fragment.endswith(u'.'):
                fragment = fragment[:-1]
                suffix = u'.' + suffix
                continue
            break
        return (fragment, prefix, suffix)

    def handle_links(self, src_iter):
        """Handle links in character tokens"""
        for token in src_iter:
            if token['type'] == 'Characters':
                text = token['data']
                new_tokens = []
                end = 0
                for match in self.url_re.finditer(text):
                    if match.start() > end:
                        new_tokens.append({u'type': u'Characters', u'data': text[end:match.start()]})
                    url = match.group(0)
                    prefix = suffix = ''
                    url, prefix, suffix = self.strip_non_url_bits(url)
                    if PROTO_RE.search(url):
                        href = url
                    else:
                        href = u'http://%s' % url
                    attrs = {(None, u'href'): href, u'_text': url}
                    attrs = self.apply_callbacks(attrs, True)
                    if attrs is None:
                        new_tokens.append({u'type': u'Characters', u'data': prefix + url + suffix})
                    else:
                        if prefix:
                            new_tokens.append({u'type': u'Characters', u'data': prefix})
                        _text = attrs.pop(u'_text', '')
                        attrs = alphabetize_attributes(attrs)
                        new_tokens.extend([{u'type': u'StartTag', u'name': u'a', u'data': attrs}, {u'type': u'Characters', u'data': force_unicode(_text)}, {u'type': u'EndTag', u'name': 'a'}])
                        if suffix:
                            new_tokens.append({u'type': u'Characters', u'data': suffix})
                    end = match.end()
                if new_tokens:
                    if end < len(text):
                        new_tokens.append({u'type': u'Characters', u'data': text[end:]})
                    for new_token in new_tokens:
                        yield new_token
                    continue
            yield token

    def handle_a_tag(self, token_buffer):
        """Handle the "a" tag

        This could adjust the link or drop it altogether depending on what the
        callbacks return.

        This yields the new set of tokens.

        """
        a_token = token_buffer[0]
        if a_token['data']:
            attrs = a_token['data']
        else:
            attrs = {}
        text = self.extract_character_data(token_buffer)
        attrs['_text'] = text
        attrs = self.apply_callbacks(attrs, False)
        if attrs is None:
            yield {'type': 'Characters', 'data': text}
        else:
            new_text = attrs.pop('_text', '')
            a_token['data'] = alphabetize_attributes(attrs)
            if text == new_text:
                yield a_token
                for mem in token_buffer[1:]:
                    yield mem
            else:
                yield a_token
                yield {'type': 'Characters', 'data': force_unicode(new_text)}
                yield token_buffer[-1]

    def __iter__(self):
        in_a = False
        in_skip_tag = None
        token_buffer = []
        for token in super(LinkifyFilter, self).__iter__():
            if in_a:
                if token['type'] == 'EndTag' and token['name'] == 'a':
                    token_buffer.append(token)
                    for new_token in self.handle_a_tag(token_buffer):
                        yield new_token
                    in_a = False
                    token_buffer = []
                    continue
                else:
                    token_buffer.append(token)
                    continue
            elif token['type'] in ['StartTag', 'EmptyTag']:
                if token['name'] in self.skip_tags:
                    in_skip_tag = token['name']
                elif token['name'] == 'a':
                    in_a = True
                    token_buffer.append(token)
                    continue
            elif in_skip_tag and self.skip_tags:
                if token['type'] == 'EndTag' and token['name'] == in_skip_tag:
                    in_skip_tag = None
            elif not in_a and (not in_skip_tag) and (token['type'] == 'Characters'):
                new_stream = iter([token])
                if self.parse_email:
                    new_stream = self.handle_email_addresses(new_stream)
                new_stream = self.handle_links(new_stream)
                for token in new_stream:
                    yield token
                continue
            yield token