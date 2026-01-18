import binascii
import json
from pymacaroons import utils
Deserialize a JSON macaroon v2.

        @param serialized the macaroon in JSON format v2.
        @return the macaroon object.
        