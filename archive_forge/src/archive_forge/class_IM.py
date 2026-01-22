from the atom and gd namespaces. For more information, see:
from __future__ import absolute_import
import base64
import calendar
import datetime
import os
import re
import time
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import namespace_manager
from googlecloudsdk.third_party.appengine.api import users
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
from googlecloudsdk.third_party.appengine.datastore import sortable_pb_encoder
from googlecloudsdk.third_party.appengine._internal import six_subset
class IM(object):
    """An instant messaging handle. Includes both an address and its protocol.
  The protocol value is either a standard IM scheme or a URL identifying the
  IM network for the protocol. Possible values include:

    Value                           Description
    sip                             SIP/SIMPLE
    unknown                         Unknown or unspecified
    xmpp                            XMPP/Jabber
    http://aim.com/                 AIM
    http://icq.com/                 ICQ
    http://talk.google.com/         Google Talk
    http://messenger.msn.com/       MSN Messenger
    http://messenger.yahoo.com/     Yahoo Messenger
    http://sametime.com/            Lotus Sametime
    http://gadu-gadu.pl/            Gadu-Gadu

  This is the gd:im element. In XML output, the address and protocol are
  provided as the address and protocol attributes, respectively. See:
  https://developers.google.com/gdata/docs/1.0/elements#gdIm

  Serializes to '<protocol> <address>'. Raises BadValueError if tag is not a
  standard IM scheme or a URL.
  """
    PROTOCOLS = ['sip', 'unknown', 'xmpp']
    protocol = None
    address = None

    def __init__(self, protocol, address=None):
        if address is None:
            try:
                split = protocol.split(' ', 1)
                protocol, address = split
            except (AttributeError, ValueError):
                raise datastore_errors.BadValueError('Expected string of format "protocol address"; received %s' % (protocol,))
        ValidateString(address, 'address')
        if protocol not in self.PROTOCOLS:
            Link(protocol)
        self.address = address
        self.protocol = protocol

    def __cmp__(self, other):
        if not isinstance(other, IM):
            try:
                other = IM(other)
            except datastore_errors.BadValueError:
                return NotImplemented
        return cmp((self.address, self.protocol), (other.address, other.protocol))

    def __repr__(self):
        """Returns an eval()able string representation of this IM.

    The returned string is of the form:

      datastore_types.IM('address', 'protocol')

    Returns:
      string
    """
        return 'datastore_types.IM(%r, %r)' % (self.protocol, self.address)

    def __unicode__(self):
        return u'%s %s' % (self.protocol, self.address)
    __str__ = __unicode__

    def ToXml(self):
        return u'<gd:im protocol=%s address=%s />' % (saxutils.quoteattr(self.protocol), saxutils.quoteattr(self.address))

    def __len__(self):
        return len(six_subset.text_type(self))