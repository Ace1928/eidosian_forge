from yowsup.structs import ProtocolEntity, ProtocolTreeNode
from .receipt import ReceiptProtocolEntity
from yowsup.layers.protocol_acks.protocolentities  import OutgoingAckProtocolEntity

    delivered:
    <receipt to="xxxxxxxxxxx@s.whatsapp.net" id="1415389947-15"></receipt>

    read
    <receipt to="xxxxxxxxxxx@s.whatsapp.net" id="1415389947-15" type="read"></receipt>

    delivered to participant in group:
    <receipt participant="xxxxxxxxxx@s.whatsapp.net" from="yyyyyyyyyyyyy@g.us" id="1431204051-9" t="1431204094"></receipt>

    read by participant in group:
    <receipt participant="xxxxxxxxxx@s.whatsapp.net" t="1431204235" from="yyyyyyyyyyyyy@g.us" id="1431204051-9" type="read"></receipt>

    multiple items:
    <receipt type="read" from="xxxxxxxxxxxx@s.whatsapp.net" id="1431364583-191" t="1431365553">
        <list>
            <item id="1431364572-189"></item>
            <item id="1431364575-190"></item>
        </list>
    </receipt>

    multiple items to group:
    <receipt participant="xxxxxxxxxxxx@s.whatsapp.net" t="1431330533" from="yyyyyyyyyyyyyy@g.us" id="1431330385-323" type="read">
        <list>
            <item id="1431330096-317"></item>
            <item id="1431330373-320"></item>
            <item id="1431330373-321"></item>
            <item id="1431330385-322"></item>
        </list>
    </receipt>

    INCOMING
    <receipt offline="0" from="xxxxxxxxxx@s.whatsapp.net" id="1415577964-1" t="1415578027"></receipt>
    