import os
import logging
from os_ken.lib import hub, alert
from os_ken.base import app_manager
from os_ken.controller import event
class EventAlert(event.EventBase):

    def __init__(self, msg):
        super(EventAlert, self).__init__()
        self.msg = msg