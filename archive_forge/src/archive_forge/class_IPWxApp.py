import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
class IPWxApp(wx.App):

    def OnInit(self):
        self.frame = TimerFrame(wake)
        self.frame.Show(False)
        return True