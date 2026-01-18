from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import ResultIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
from axolotl.state.prekeybundle import PreKeyBundle
from axolotl.identitykey import IdentityKey
from axolotl.ecc.curve import Curve
from axolotl.ecc.djbec import DjbECPublicKey
import binascii
import sys

    <iq type="result" from="s.whatsapp.net" id="3">
    <list>
    <user jid="79049347231@s.whatsapp.net">
    <registration>
        HEX:7a9cec4b</registration>
    <type>

    HEX:05</type>
    <identity>
    HEX:eeb668c8d062c99b43560c811acfe6e492798b496767eb060d99e011d3862369</identity>
    <skey>
    <id>

    HEX:000000</id>
    <value>
    HEX:a1b5216ce4678143fb20aaaa2711a8c2b647230164b79414f0550b4e611ccd6c</value>
    <signature>
    HEX:94c231327fcd664b34603838b5e9ba926718d71c206e92b2b400f5cf4ae7bf17d83557bf328c1be6d51efdbd731a26d000adb8f38f140b1ea2a5fd3df2688085</signature>
        </skey>
    <key>
        <id>
        HEX:36b545</id>
    <value>
    HEX:c20826f622bec24b349ced38f1854bdec89ba098ef4c06b2402800d33e9aff61</value>
    </key>
    </user>
    </list>
    </iq>
    