from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class GetOAuthUserRequest(ProtocolBuffer.ProtocolMessage):
    has_scope_ = 0
    scope_ = ''
    has_request_writer_permission_ = 0
    request_writer_permission_ = 0

    def __init__(self, contents=None):
        self.scopes_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def scope(self):
        return self.scope_

    def set_scope(self, x):
        self.has_scope_ = 1
        self.scope_ = x

    def clear_scope(self):
        if self.has_scope_:
            self.has_scope_ = 0
            self.scope_ = ''

    def has_scope(self):
        return self.has_scope_

    def scopes_size(self):
        return len(self.scopes_)

    def scopes_list(self):
        return self.scopes_

    def scopes(self, i):
        return self.scopes_[i]

    def set_scopes(self, i, x):
        self.scopes_[i] = x

    def add_scopes(self, x):
        self.scopes_.append(x)

    def clear_scopes(self):
        self.scopes_ = []

    def request_writer_permission(self):
        return self.request_writer_permission_

    def set_request_writer_permission(self, x):
        self.has_request_writer_permission_ = 1
        self.request_writer_permission_ = x

    def clear_request_writer_permission(self):
        if self.has_request_writer_permission_:
            self.has_request_writer_permission_ = 0
            self.request_writer_permission_ = 0

    def has_request_writer_permission(self):
        return self.has_request_writer_permission_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_scope():
            self.set_scope(x.scope())
        for i in range(x.scopes_size()):
            self.add_scopes(x.scopes(i))
        if x.has_request_writer_permission():
            self.set_request_writer_permission(x.request_writer_permission())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_scope_ != x.has_scope_:
            return 0
        if self.has_scope_ and self.scope_ != x.scope_:
            return 0
        if len(self.scopes_) != len(x.scopes_):
            return 0
        for e1, e2 in zip(self.scopes_, x.scopes_):
            if e1 != e2:
                return 0
        if self.has_request_writer_permission_ != x.has_request_writer_permission_:
            return 0
        if self.has_request_writer_permission_ and self.request_writer_permission_ != x.request_writer_permission_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_scope_:
            n += 1 + self.lengthString(len(self.scope_))
        n += 1 * len(self.scopes_)
        for i in range(len(self.scopes_)):
            n += self.lengthString(len(self.scopes_[i]))
        if self.has_request_writer_permission_:
            n += 2
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_scope_:
            n += 1 + self.lengthString(len(self.scope_))
        n += 1 * len(self.scopes_)
        for i in range(len(self.scopes_)):
            n += self.lengthString(len(self.scopes_[i]))
        if self.has_request_writer_permission_:
            n += 2
        return n

    def Clear(self):
        self.clear_scope()
        self.clear_scopes()
        self.clear_request_writer_permission()

    def OutputUnchecked(self, out):
        if self.has_scope_:
            out.putVarInt32(10)
            out.putPrefixedString(self.scope_)
        for i in range(len(self.scopes_)):
            out.putVarInt32(18)
            out.putPrefixedString(self.scopes_[i])
        if self.has_request_writer_permission_:
            out.putVarInt32(24)
            out.putBoolean(self.request_writer_permission_)

    def OutputPartial(self, out):
        if self.has_scope_:
            out.putVarInt32(10)
            out.putPrefixedString(self.scope_)
        for i in range(len(self.scopes_)):
            out.putVarInt32(18)
            out.putPrefixedString(self.scopes_[i])
        if self.has_request_writer_permission_:
            out.putVarInt32(24)
            out.putBoolean(self.request_writer_permission_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_scope(d.getPrefixedString())
                continue
            if tt == 18:
                self.add_scopes(d.getPrefixedString())
                continue
            if tt == 24:
                self.set_request_writer_permission(d.getBoolean())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_scope_:
            res += prefix + 'scope: %s\n' % self.DebugFormatString(self.scope_)
        cnt = 0
        for e in self.scopes_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'scopes%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        if self.has_request_writer_permission_:
            res += prefix + 'request_writer_permission: %s\n' % self.DebugFormatBool(self.request_writer_permission_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kscope = 1
    kscopes = 2
    krequest_writer_permission = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'scope', 2: 'scopes', 3: 'request_writer_permission'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.GetOAuthUserRequest'