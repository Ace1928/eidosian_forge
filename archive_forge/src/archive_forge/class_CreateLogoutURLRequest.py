from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class CreateLogoutURLRequest(ProtocolBuffer.ProtocolMessage):
    has_destination_url_ = 0
    destination_url_ = ''
    has_auth_domain_ = 0
    auth_domain_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def destination_url(self):
        return self.destination_url_

    def set_destination_url(self, x):
        self.has_destination_url_ = 1
        self.destination_url_ = x

    def clear_destination_url(self):
        if self.has_destination_url_:
            self.has_destination_url_ = 0
            self.destination_url_ = ''

    def has_destination_url(self):
        return self.has_destination_url_

    def auth_domain(self):
        return self.auth_domain_

    def set_auth_domain(self, x):
        self.has_auth_domain_ = 1
        self.auth_domain_ = x

    def clear_auth_domain(self):
        if self.has_auth_domain_:
            self.has_auth_domain_ = 0
            self.auth_domain_ = ''

    def has_auth_domain(self):
        return self.has_auth_domain_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_destination_url():
            self.set_destination_url(x.destination_url())
        if x.has_auth_domain():
            self.set_auth_domain(x.auth_domain())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_destination_url_ != x.has_destination_url_:
            return 0
        if self.has_destination_url_ and self.destination_url_ != x.destination_url_:
            return 0
        if self.has_auth_domain_ != x.has_auth_domain_:
            return 0
        if self.has_auth_domain_ and self.auth_domain_ != x.auth_domain_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_destination_url_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: destination_url not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.destination_url_))
        if self.has_auth_domain_:
            n += 1 + self.lengthString(len(self.auth_domain_))
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_destination_url_:
            n += 1
            n += self.lengthString(len(self.destination_url_))
        if self.has_auth_domain_:
            n += 1 + self.lengthString(len(self.auth_domain_))
        return n

    def Clear(self):
        self.clear_destination_url()
        self.clear_auth_domain()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.destination_url_)
        if self.has_auth_domain_:
            out.putVarInt32(18)
            out.putPrefixedString(self.auth_domain_)

    def OutputPartial(self, out):
        if self.has_destination_url_:
            out.putVarInt32(10)
            out.putPrefixedString(self.destination_url_)
        if self.has_auth_domain_:
            out.putVarInt32(18)
            out.putPrefixedString(self.auth_domain_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_destination_url(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_auth_domain(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_destination_url_:
            res += prefix + 'destination_url: %s\n' % self.DebugFormatString(self.destination_url_)
        if self.has_auth_domain_:
            res += prefix + 'auth_domain: %s\n' % self.DebugFormatString(self.auth_domain_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kdestination_url = 1
    kauth_domain = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'destination_url', 2: 'auth_domain'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.CreateLogoutURLRequest'