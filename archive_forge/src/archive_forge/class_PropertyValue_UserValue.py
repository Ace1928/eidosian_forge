from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class PropertyValue_UserValue(ProtocolBuffer.ProtocolMessage):
    has_email_ = 0
    email_ = ''
    has_auth_domain_ = 0
    auth_domain_ = ''
    has_nickname_ = 0
    nickname_ = ''
    has_gaiaid_ = 0
    gaiaid_ = 0
    has_obfuscated_gaiaid_ = 0
    obfuscated_gaiaid_ = ''
    has_federated_identity_ = 0
    federated_identity_ = ''
    has_federated_provider_ = 0
    federated_provider_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def email(self):
        return self.email_

    def set_email(self, x):
        self.has_email_ = 1
        self.email_ = x

    def clear_email(self):
        if self.has_email_:
            self.has_email_ = 0
            self.email_ = ''

    def has_email(self):
        return self.has_email_

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

    def nickname(self):
        return self.nickname_

    def set_nickname(self, x):
        self.has_nickname_ = 1
        self.nickname_ = x

    def clear_nickname(self):
        if self.has_nickname_:
            self.has_nickname_ = 0
            self.nickname_ = ''

    def has_nickname(self):
        return self.has_nickname_

    def gaiaid(self):
        return self.gaiaid_

    def set_gaiaid(self, x):
        self.has_gaiaid_ = 1
        self.gaiaid_ = x

    def clear_gaiaid(self):
        if self.has_gaiaid_:
            self.has_gaiaid_ = 0
            self.gaiaid_ = 0

    def has_gaiaid(self):
        return self.has_gaiaid_

    def obfuscated_gaiaid(self):
        return self.obfuscated_gaiaid_

    def set_obfuscated_gaiaid(self, x):
        self.has_obfuscated_gaiaid_ = 1
        self.obfuscated_gaiaid_ = x

    def clear_obfuscated_gaiaid(self):
        if self.has_obfuscated_gaiaid_:
            self.has_obfuscated_gaiaid_ = 0
            self.obfuscated_gaiaid_ = ''

    def has_obfuscated_gaiaid(self):
        return self.has_obfuscated_gaiaid_

    def federated_identity(self):
        return self.federated_identity_

    def set_federated_identity(self, x):
        self.has_federated_identity_ = 1
        self.federated_identity_ = x

    def clear_federated_identity(self):
        if self.has_federated_identity_:
            self.has_federated_identity_ = 0
            self.federated_identity_ = ''

    def has_federated_identity(self):
        return self.has_federated_identity_

    def federated_provider(self):
        return self.federated_provider_

    def set_federated_provider(self, x):
        self.has_federated_provider_ = 1
        self.federated_provider_ = x

    def clear_federated_provider(self):
        if self.has_federated_provider_:
            self.has_federated_provider_ = 0
            self.federated_provider_ = ''

    def has_federated_provider(self):
        return self.has_federated_provider_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_email():
            self.set_email(x.email())
        if x.has_auth_domain():
            self.set_auth_domain(x.auth_domain())
        if x.has_nickname():
            self.set_nickname(x.nickname())
        if x.has_gaiaid():
            self.set_gaiaid(x.gaiaid())
        if x.has_obfuscated_gaiaid():
            self.set_obfuscated_gaiaid(x.obfuscated_gaiaid())
        if x.has_federated_identity():
            self.set_federated_identity(x.federated_identity())
        if x.has_federated_provider():
            self.set_federated_provider(x.federated_provider())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_email_ != x.has_email_:
            return 0
        if self.has_email_ and self.email_ != x.email_:
            return 0
        if self.has_auth_domain_ != x.has_auth_domain_:
            return 0
        if self.has_auth_domain_ and self.auth_domain_ != x.auth_domain_:
            return 0
        if self.has_nickname_ != x.has_nickname_:
            return 0
        if self.has_nickname_ and self.nickname_ != x.nickname_:
            return 0
        if self.has_gaiaid_ != x.has_gaiaid_:
            return 0
        if self.has_gaiaid_ and self.gaiaid_ != x.gaiaid_:
            return 0
        if self.has_obfuscated_gaiaid_ != x.has_obfuscated_gaiaid_:
            return 0
        if self.has_obfuscated_gaiaid_ and self.obfuscated_gaiaid_ != x.obfuscated_gaiaid_:
            return 0
        if self.has_federated_identity_ != x.has_federated_identity_:
            return 0
        if self.has_federated_identity_ and self.federated_identity_ != x.federated_identity_:
            return 0
        if self.has_federated_provider_ != x.has_federated_provider_:
            return 0
        if self.has_federated_provider_ and self.federated_provider_ != x.federated_provider_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_email_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: email not set.')
        if not self.has_auth_domain_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: auth_domain not set.')
        if not self.has_gaiaid_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: gaiaid not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.email_))
        n += self.lengthString(len(self.auth_domain_))
        if self.has_nickname_:
            n += 1 + self.lengthString(len(self.nickname_))
        n += self.lengthVarInt64(self.gaiaid_)
        if self.has_obfuscated_gaiaid_:
            n += 2 + self.lengthString(len(self.obfuscated_gaiaid_))
        if self.has_federated_identity_:
            n += 2 + self.lengthString(len(self.federated_identity_))
        if self.has_federated_provider_:
            n += 2 + self.lengthString(len(self.federated_provider_))
        return n + 4

    def ByteSizePartial(self):
        n = 0
        if self.has_email_:
            n += 1
            n += self.lengthString(len(self.email_))
        if self.has_auth_domain_:
            n += 1
            n += self.lengthString(len(self.auth_domain_))
        if self.has_nickname_:
            n += 1 + self.lengthString(len(self.nickname_))
        if self.has_gaiaid_:
            n += 2
            n += self.lengthVarInt64(self.gaiaid_)
        if self.has_obfuscated_gaiaid_:
            n += 2 + self.lengthString(len(self.obfuscated_gaiaid_))
        if self.has_federated_identity_:
            n += 2 + self.lengthString(len(self.federated_identity_))
        if self.has_federated_provider_:
            n += 2 + self.lengthString(len(self.federated_provider_))
        return n

    def Clear(self):
        self.clear_email()
        self.clear_auth_domain()
        self.clear_nickname()
        self.clear_gaiaid()
        self.clear_obfuscated_gaiaid()
        self.clear_federated_identity()
        self.clear_federated_provider()

    def OutputUnchecked(self, out):
        out.putVarInt32(74)
        out.putPrefixedString(self.email_)
        out.putVarInt32(82)
        out.putPrefixedString(self.auth_domain_)
        if self.has_nickname_:
            out.putVarInt32(90)
            out.putPrefixedString(self.nickname_)
        out.putVarInt32(144)
        out.putVarInt64(self.gaiaid_)
        if self.has_obfuscated_gaiaid_:
            out.putVarInt32(154)
            out.putPrefixedString(self.obfuscated_gaiaid_)
        if self.has_federated_identity_:
            out.putVarInt32(170)
            out.putPrefixedString(self.federated_identity_)
        if self.has_federated_provider_:
            out.putVarInt32(178)
            out.putPrefixedString(self.federated_provider_)

    def OutputPartial(self, out):
        if self.has_email_:
            out.putVarInt32(74)
            out.putPrefixedString(self.email_)
        if self.has_auth_domain_:
            out.putVarInt32(82)
            out.putPrefixedString(self.auth_domain_)
        if self.has_nickname_:
            out.putVarInt32(90)
            out.putPrefixedString(self.nickname_)
        if self.has_gaiaid_:
            out.putVarInt32(144)
            out.putVarInt64(self.gaiaid_)
        if self.has_obfuscated_gaiaid_:
            out.putVarInt32(154)
            out.putPrefixedString(self.obfuscated_gaiaid_)
        if self.has_federated_identity_:
            out.putVarInt32(170)
            out.putPrefixedString(self.federated_identity_)
        if self.has_federated_provider_:
            out.putVarInt32(178)
            out.putPrefixedString(self.federated_provider_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 68:
                break
            if tt == 74:
                self.set_email(d.getPrefixedString())
                continue
            if tt == 82:
                self.set_auth_domain(d.getPrefixedString())
                continue
            if tt == 90:
                self.set_nickname(d.getPrefixedString())
                continue
            if tt == 144:
                self.set_gaiaid(d.getVarInt64())
                continue
            if tt == 154:
                self.set_obfuscated_gaiaid(d.getPrefixedString())
                continue
            if tt == 170:
                self.set_federated_identity(d.getPrefixedString())
                continue
            if tt == 178:
                self.set_federated_provider(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_email_:
            res += prefix + 'email: %s\n' % self.DebugFormatString(self.email_)
        if self.has_auth_domain_:
            res += prefix + 'auth_domain: %s\n' % self.DebugFormatString(self.auth_domain_)
        if self.has_nickname_:
            res += prefix + 'nickname: %s\n' % self.DebugFormatString(self.nickname_)
        if self.has_gaiaid_:
            res += prefix + 'gaiaid: %s\n' % self.DebugFormatInt64(self.gaiaid_)
        if self.has_obfuscated_gaiaid_:
            res += prefix + 'obfuscated_gaiaid: %s\n' % self.DebugFormatString(self.obfuscated_gaiaid_)
        if self.has_federated_identity_:
            res += prefix + 'federated_identity: %s\n' % self.DebugFormatString(self.federated_identity_)
        if self.has_federated_provider_:
            res += prefix + 'federated_provider: %s\n' % self.DebugFormatString(self.federated_provider_)
        return res