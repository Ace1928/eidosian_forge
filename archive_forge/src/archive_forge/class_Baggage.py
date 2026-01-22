import contextlib
import os
import re
import sys
import sentry_sdk
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.utils import (
from sentry_sdk._compat import PY2, duration_in_milliseconds, iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing import LOW_QUALITY_TRANSACTION_SOURCES
class Baggage(object):
    """
    The W3C Baggage header information (see https://www.w3.org/TR/baggage/).
    """
    __slots__ = ('sentry_items', 'third_party_items', 'mutable')
    SENTRY_PREFIX = 'sentry-'
    SENTRY_PREFIX_REGEX = re.compile('^sentry-')

    def __init__(self, sentry_items, third_party_items='', mutable=True):
        self.sentry_items = sentry_items
        self.third_party_items = third_party_items
        self.mutable = mutable

    @classmethod
    def from_incoming_header(cls, header):
        """
        freeze if incoming header already has sentry baggage
        """
        sentry_items = {}
        third_party_items = ''
        mutable = True
        if header:
            for item in header.split(','):
                if '=' not in item:
                    continue
                with capture_internal_exceptions():
                    item = item.strip()
                    key, val = item.split('=')
                    if Baggage.SENTRY_PREFIX_REGEX.match(key):
                        baggage_key = unquote(key.split('-')[1])
                        sentry_items[baggage_key] = unquote(val)
                        mutable = False
                    else:
                        third_party_items += (',' if third_party_items else '') + item
        return Baggage(sentry_items, third_party_items, mutable)

    @classmethod
    def from_options(cls, scope):
        sentry_items = {}
        third_party_items = ''
        mutable = False
        client = sentry_sdk.Hub.current.client
        if client is None or scope._propagation_context is None:
            return Baggage(sentry_items)
        options = client.options
        propagation_context = scope._propagation_context
        if propagation_context is not None and 'trace_id' in propagation_context:
            sentry_items['trace_id'] = propagation_context['trace_id']
        if options.get('environment'):
            sentry_items['environment'] = options['environment']
        if options.get('release'):
            sentry_items['release'] = options['release']
        if options.get('dsn'):
            sentry_items['public_key'] = Dsn(options['dsn']).public_key
        if options.get('traces_sample_rate'):
            sentry_items['sample_rate'] = options['traces_sample_rate']
        user = scope and scope._user or {}
        if user.get('segment'):
            sentry_items['user_segment'] = user['segment']
        return Baggage(sentry_items, third_party_items, mutable)

    @classmethod
    def populate_from_transaction(cls, transaction):
        """
        Populate fresh baggage entry with sentry_items and make it immutable
        if this is the head SDK which originates traces.
        """
        hub = transaction.hub or sentry_sdk.Hub.current
        client = hub.client
        sentry_items = {}
        if not client:
            return Baggage(sentry_items)
        options = client.options or {}
        user = hub.scope and hub.scope._user or {}
        sentry_items['trace_id'] = transaction.trace_id
        if options.get('environment'):
            sentry_items['environment'] = options['environment']
        if options.get('release'):
            sentry_items['release'] = options['release']
        if options.get('dsn'):
            sentry_items['public_key'] = Dsn(options['dsn']).public_key
        if transaction.name and transaction.source not in LOW_QUALITY_TRANSACTION_SOURCES:
            sentry_items['transaction'] = transaction.name
        if user.get('segment'):
            sentry_items['user_segment'] = user['segment']
        if transaction.sample_rate is not None:
            sentry_items['sample_rate'] = str(transaction.sample_rate)
        if transaction.sampled is not None:
            sentry_items['sampled'] = 'true' if transaction.sampled else 'false'
        if transaction._baggage and transaction._baggage.sentry_items:
            sentry_items.update(transaction._baggage.sentry_items)
        return Baggage(sentry_items, mutable=False)

    def freeze(self):
        self.mutable = False

    def dynamic_sampling_context(self):
        header = {}
        for key, item in iteritems(self.sentry_items):
            header[key] = item
        return header

    def serialize(self, include_third_party=False):
        items = []
        for key, val in iteritems(self.sentry_items):
            with capture_internal_exceptions():
                item = Baggage.SENTRY_PREFIX + quote(key) + '=' + quote(str(val))
                items.append(item)
        if include_third_party:
            items.append(self.third_party_items)
        return ','.join(items)