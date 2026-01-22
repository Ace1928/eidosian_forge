from collections import namedtuple
import macaroonbakery.checkers as checkers
class ThirdPartyCaveatInfo(namedtuple('ThirdPartyCaveatInfo', 'condition, first_party_public_key, third_party_key_pair, root_key, caveat, version, id, namespace')):
    """ThirdPartyCaveatInfo holds the information decoded from
    a third party caveat id.

    @param condition holds the third party condition to be discharged.
    This is the only field that most third party dischargers will
    need to consider. {str}

    @param first_party_public_key holds the public key of the party
    that created the third party caveat. {PublicKey}

    @param third_party_key_pair holds the nacl private used to decrypt
    the caveat - the key pair of the discharging service. {PrivateKey}

    @param root_key holds the secret root key encoded by the caveat. {bytes}

    @param caveat holds the full caveat id from
    which all the other fields are derived. {bytes}

    @param version holds the version that was used to encode
    the caveat id. {number}

    @param id holds the id of the third party caveat (the id that the
    discharge macaroon should be given). This will differ from Caveat
    when the caveat information is encoded separately. {bytes}

    @param namespace object that holds the namespace of the first party
    that created the macaroon, as encoded by the party that added the
    third party caveat. {checkers.Namespace}
    """