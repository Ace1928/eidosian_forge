from yowsup.structs import ProtocolEntity, ProtocolTreeNode

    <presence type="{{type}} name={{push_name}}"></presence>
    Should normally be either type or name

    when contact goes offline:
    <presence type="unavailable" from="{{contact_jid}}" last="deny | ?">
    </presence>

    when contact goes online:
    <presence from="contact_jid">
    </presence>

    