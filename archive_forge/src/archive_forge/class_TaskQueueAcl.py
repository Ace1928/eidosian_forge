from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueAcl(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.user_email_ = []
        self.writer_email_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def user_email_size(self):
        return len(self.user_email_)

    def user_email_list(self):
        return self.user_email_

    def user_email(self, i):
        return self.user_email_[i]

    def set_user_email(self, i, x):
        self.user_email_[i] = x

    def add_user_email(self, x):
        self.user_email_.append(x)

    def clear_user_email(self):
        self.user_email_ = []

    def writer_email_size(self):
        return len(self.writer_email_)

    def writer_email_list(self):
        return self.writer_email_

    def writer_email(self, i):
        return self.writer_email_[i]

    def set_writer_email(self, i, x):
        self.writer_email_[i] = x

    def add_writer_email(self, x):
        self.writer_email_.append(x)

    def clear_writer_email(self):
        self.writer_email_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.user_email_size()):
            self.add_user_email(x.user_email(i))
        for i in range(x.writer_email_size()):
            self.add_writer_email(x.writer_email(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.user_email_) != len(x.user_email_):
            return 0
        for e1, e2 in zip(self.user_email_, x.user_email_):
            if e1 != e2:
                return 0
        if len(self.writer_email_) != len(x.writer_email_):
            return 0
        for e1, e2 in zip(self.writer_email_, x.writer_email_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.user_email_)
        for i in range(len(self.user_email_)):
            n += self.lengthString(len(self.user_email_[i]))
        n += 1 * len(self.writer_email_)
        for i in range(len(self.writer_email_)):
            n += self.lengthString(len(self.writer_email_[i]))
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.user_email_)
        for i in range(len(self.user_email_)):
            n += self.lengthString(len(self.user_email_[i]))
        n += 1 * len(self.writer_email_)
        for i in range(len(self.writer_email_)):
            n += self.lengthString(len(self.writer_email_[i]))
        return n

    def Clear(self):
        self.clear_user_email()
        self.clear_writer_email()

    def OutputUnchecked(self, out):
        for i in range(len(self.user_email_)):
            out.putVarInt32(10)
            out.putPrefixedString(self.user_email_[i])
        for i in range(len(self.writer_email_)):
            out.putVarInt32(18)
            out.putPrefixedString(self.writer_email_[i])

    def OutputPartial(self, out):
        for i in range(len(self.user_email_)):
            out.putVarInt32(10)
            out.putPrefixedString(self.user_email_[i])
        for i in range(len(self.writer_email_)):
            out.putVarInt32(18)
            out.putPrefixedString(self.writer_email_[i])

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.add_user_email(d.getPrefixedString())
                continue
            if tt == 18:
                self.add_writer_email(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.user_email_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'user_email%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        cnt = 0
        for e in self.writer_email_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'writer_email%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kuser_email = 1
    kwriter_email = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'user_email', 2: 'writer_email'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueAcl'