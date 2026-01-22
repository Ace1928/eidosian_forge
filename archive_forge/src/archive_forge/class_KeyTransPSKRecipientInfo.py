from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5652
class KeyTransPSKRecipientInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', rfc5652.CMSVersion()), namedtype.NamedType('pskid', PreSharedKeyIdentifier()), namedtype.NamedType('kdfAlgorithm', rfc5652.KeyDerivationAlgorithmIdentifier()), namedtype.NamedType('keyEncryptionAlgorithm', rfc5652.KeyEncryptionAlgorithmIdentifier()), namedtype.NamedType('ktris', KeyTransRecipientInfos()), namedtype.NamedType('encryptedKey', rfc5652.EncryptedKey()))