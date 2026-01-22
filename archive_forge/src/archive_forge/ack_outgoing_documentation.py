from .ack import AckProtocolEntity

    <ack type="{{delivery | read}}" class="{{message | receipt | ?}}" id="{{MESSAGE_ID}} to={{TO_JID}}">
    </ack>

    <ack to="{{GROUP_JID}}" participant="{{JID}}" id="{{MESSAGE_ID}}" class="receipt" type="{{read | }}">
    </ack>

    