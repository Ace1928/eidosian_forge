import logging
import os
from parlai.core.agents import create_agent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.server as server_utils
from parlai.chat_service.services.messenger.agents import MessengerAgent
from parlai.chat_service.core.socket import ChatServiceMessageSocket
from parlai.chat_service.services.messenger.message_sender import MessageSender
from parlai.chat_service.core.chat_service_manager import ChatServiceManager
def _handle_webhook_event(self, event):
    if 'message' in event:
        if 'image_url' in event and event['image_url'] is not None or ('attachment_url' in event and event['attachment_url'] is not None):
            event['message']['image'] = True
        self._on_new_message(event)
    elif 'delivery' in event:
        self.confirm_message_delivery(event)
    elif 'read' in event:
        self.handle_message_read(event)