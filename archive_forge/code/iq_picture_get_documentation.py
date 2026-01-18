from .iq_picture import PictureIqProtocolEntity
from yowsup.structs import ProtocolTreeNode

    <iq type="get" id="{{id}}" xmlns="w:profile:picture", to={{jid}}">
        <picture type="image | preview">
        </picture>
    </iq>