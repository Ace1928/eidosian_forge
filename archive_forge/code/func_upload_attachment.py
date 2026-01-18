import logging
import os
from parlai.core.agents import create_agent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.server as server_utils
from parlai.chat_service.services.messenger.agents import MessengerAgent
from parlai.chat_service.core.socket import ChatServiceMessageSocket
from parlai.chat_service.services.messenger.message_sender import MessageSender
from parlai.chat_service.core.chat_service_manager import ChatServiceManager
def upload_attachment(self, payload):
    """
        Upload an attachment and return an attachment ID.

        :param payload:
            dict with the following format:
                {'type': <TYPE>, 'url': <URL>} or
                {'type': <TYPE>, 'filename': <FILENAME>, 'format': <FILEFORMAT>}.
                For example,
                {'type': 'image', 'filename': 'test.png', 'format': 'png'}
        """
    return self.sender.upload_fb_attachment(payload)