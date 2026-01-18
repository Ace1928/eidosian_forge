from importlib import reload
import unittest
import logging
def test_ofp_event(self):
    import os_ken.ofproto
    reload(os_ken.ofproto)
    import os_ken.controller.ofp_event
    reload(os_ken.controller.ofp_event)