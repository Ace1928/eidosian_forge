import binascii
import json
from pymacaroons import utils
def _deserialize_v2(self, deserialized):
    """Deserialize a JSON macaroon v2.

        @param serialized the macaroon in JSON format v2.
        @return the macaroon object.
        """
    from pymacaroons.macaroon import Macaroon, MACAROON_V2
    from pymacaroons.caveat import Caveat
    caveats = []
    for c in deserialized.get('c', []):
        caveat = Caveat(caveat_id=_read_json_binary_field(c, 'i'), verification_key_id=_read_json_binary_field(c, 'v'), location=_read_json_binary_field(c, 'l'), version=MACAROON_V2)
        caveats.append(caveat)
    return Macaroon(location=_read_json_binary_field(deserialized, 'l'), identifier=_read_json_binary_field(deserialized, 'i'), caveats=caveats, signature=binascii.hexlify(_read_json_binary_field(deserialized, 's')), version=MACAROON_V2)