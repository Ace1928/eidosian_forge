from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb
class NextRequest(ProtocolBuffer.ProtocolMessage):
    has_cursor_ = 0
    has_count_ = 0
    count_ = 0
    has_offset_ = 0
    offset_ = 0
    has_compile_ = 0
    compile_ = 0

    def __init__(self, contents=None):
        self.cursor_ = Cursor()
        if contents is not None:
            self.MergeFromString(contents)

    def cursor(self):
        return self.cursor_

    def mutable_cursor(self):
        self.has_cursor_ = 1
        return self.cursor_

    def clear_cursor(self):
        self.has_cursor_ = 0
        self.cursor_.Clear()

    def has_cursor(self):
        return self.has_cursor_

    def count(self):
        return self.count_

    def set_count(self, x):
        self.has_count_ = 1
        self.count_ = x

    def clear_count(self):
        if self.has_count_:
            self.has_count_ = 0
            self.count_ = 0

    def has_count(self):
        return self.has_count_

    def offset(self):
        return self.offset_

    def set_offset(self, x):
        self.has_offset_ = 1
        self.offset_ = x

    def clear_offset(self):
        if self.has_offset_:
            self.has_offset_ = 0
            self.offset_ = 0

    def has_offset(self):
        return self.has_offset_

    def compile(self):
        return self.compile_

    def set_compile(self, x):
        self.has_compile_ = 1
        self.compile_ = x

    def clear_compile(self):
        if self.has_compile_:
            self.has_compile_ = 0
            self.compile_ = 0

    def has_compile(self):
        return self.has_compile_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_cursor():
            self.mutable_cursor().MergeFrom(x.cursor())
        if x.has_count():
            self.set_count(x.count())
        if x.has_offset():
            self.set_offset(x.offset())
        if x.has_compile():
            self.set_compile(x.compile())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_cursor_ != x.has_cursor_:
            return 0
        if self.has_cursor_ and self.cursor_ != x.cursor_:
            return 0
        if self.has_count_ != x.has_count_:
            return 0
        if self.has_count_ and self.count_ != x.count_:
            return 0
        if self.has_offset_ != x.has_offset_:
            return 0
        if self.has_offset_ and self.offset_ != x.offset_:
            return 0
        if self.has_compile_ != x.has_compile_:
            return 0
        if self.has_compile_ and self.compile_ != x.compile_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_cursor_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: cursor not set.')
        elif not self.cursor_.IsInitialized(debug_strs):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(self.cursor_.ByteSize())
        if self.has_count_:
            n += 1 + self.lengthVarInt64(self.count_)
        if self.has_offset_:
            n += 1 + self.lengthVarInt64(self.offset_)
        if self.has_compile_:
            n += 2
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_cursor_:
            n += 1
            n += self.lengthString(self.cursor_.ByteSizePartial())
        if self.has_count_:
            n += 1 + self.lengthVarInt64(self.count_)
        if self.has_offset_:
            n += 1 + self.lengthVarInt64(self.offset_)
        if self.has_compile_:
            n += 2
        return n

    def Clear(self):
        self.clear_cursor()
        self.clear_count()
        self.clear_offset()
        self.clear_compile()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putVarInt32(self.cursor_.ByteSize())
        self.cursor_.OutputUnchecked(out)
        if self.has_count_:
            out.putVarInt32(16)
            out.putVarInt32(self.count_)
        if self.has_compile_:
            out.putVarInt32(24)
            out.putBoolean(self.compile_)
        if self.has_offset_:
            out.putVarInt32(32)
            out.putVarInt32(self.offset_)

    def OutputPartial(self, out):
        if self.has_cursor_:
            out.putVarInt32(10)
            out.putVarInt32(self.cursor_.ByteSizePartial())
            self.cursor_.OutputPartial(out)
        if self.has_count_:
            out.putVarInt32(16)
            out.putVarInt32(self.count_)
        if self.has_compile_:
            out.putVarInt32(24)
            out.putBoolean(self.compile_)
        if self.has_offset_:
            out.putVarInt32(32)
            out.putVarInt32(self.offset_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_cursor().TryMerge(tmp)
                continue
            if tt == 16:
                self.set_count(d.getVarInt32())
                continue
            if tt == 24:
                self.set_compile(d.getBoolean())
                continue
            if tt == 32:
                self.set_offset(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_cursor_:
            res += prefix + 'cursor <\n'
            res += self.cursor_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_count_:
            res += prefix + 'count: %s\n' % self.DebugFormatInt32(self.count_)
        if self.has_offset_:
            res += prefix + 'offset: %s\n' % self.DebugFormatInt32(self.offset_)
        if self.has_compile_:
            res += prefix + 'compile: %s\n' % self.DebugFormatBool(self.compile_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kcursor = 1
    kcount = 2
    koffset = 4
    kcompile = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'cursor', 2: 'count', 3: 'compile', 4: 'offset'}, 4)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.NUMERIC}, 4, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.NextRequest'